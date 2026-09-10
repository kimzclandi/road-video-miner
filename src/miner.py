"""Temporal clip coverage selection and reference evaluation, independent of labels."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np


def digest(x):
    return hashlib.sha256(
        json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def immutable(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if read(p) != x:
            raise ValueError(f"Immutable evidence drift: {p}")
    else:
        # Exclusive creation; a failed partial write will be rejected, never silently reused.
        with p.open("x") as f:
            json.dump(x, f, indent=2, allow_nan=False)
            f.write("\n")


def check_clips(clips):
    seen = set()
    groups = {}
    for c in clips:
        if (
            c["id"] in seen
            or c["seconds"] != 1
            or c["fps"] != 10
            or c["end_frame_exclusive"] - c["start_frame"] != 10
        ):
            raise ValueError("Invalid clip identity or time budget")
        seen.add(c["id"])
        groups.setdefault(c["sequence"], []).append(c)
        if len(c["frames"]) != 5 or len(set(c["frames"])) != 5:
            raise ValueError("Missing/duplicate observations")
        expected = [
            f"training/image_02/{c['sequence']}/{i:06d}.png"
            for i in range(c["start_frame"], c["end_frame_exclusive"], 2)
        ]
        if c["frames"] != expected:
            raise ValueError("Frame lineage or timing mismatch")
    for cs in groups.values():
        cs = sorted(cs, key=lambda x: x["start_frame"])
        if len({c["split"] for c in cs}) != 1:
            raise ValueError("Sequence leakage")
        if any(a["end_frame_exclusive"] > b["start_frame"] for a, b in zip(cs, cs[1:])):
            raise ValueError("Overlapping clips")


def unit(x):
    x = np.asarray(x, dtype=float)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def signatures(clips, frames):
    out = []
    for c in clips:
        a = np.asarray([frames[i]["appearance"] for i in c["frames"]])
        # Appearance mean plus temporal absolute differences; 3x8-bin RGB histograms.
        appearance = unit(a.mean(axis=0))
        change = unit(np.abs(np.diff(a, axis=0)).mean(axis=0))
        out.append(
            {
                "id": c["id"],
                "appearance": appearance.tolist(),
                "temporal": unit(np.r_[appearance, change]).tolist(),
            }
        )
    return out


def choose(items, method, seed, budget=2):
    if any(set(i) != {"id", "appearance", "temporal"} for i in items):
        raise ValueError("Unlabeled signature contract")
    if not 0 < budget <= len(items):
        raise ValueError("Invalid budget")
    rng = random.Random(seed)
    n = len(items)
    if method == "random":
        return rng.sample(range(n), budget)
    if method == "uniform":
        return [int((i + 0.5) * n / budget) for i in range(budget)]
    if method not in ("dedup", "appearance_facility", "temporal"):
        raise ValueError("Unknown policy")
    x = np.asarray([r["temporal" if method == "temporal" else "appearance"] for r in items])
    sim = np.clip(x @ x.T, 0, 1)
    chosen = []
    coverage = np.zeros(n)
    tie = list(range(n))
    rng.shuffle(tie)
    for _ in range(budget):
        candidates = [i for i in tie if i not in chosen]
        if method == "dedup":
            i = max(candidates, key=lambda j: 1.0 - max((sim[j, k] for k in chosen), default=0.0))
        else:
            i = max(candidates, key=lambda j: float(np.maximum(coverage, sim[:, j]).sum() - coverage.sum()))
        chosen.append(i)
        coverage = np.maximum(coverage, sim[:, i])
    return chosen


def iou(a, b):
    v = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - v
    return v / u if u > 0 else 0.0


def reference(lines):
    out = {}
    for line in lines.splitlines():
        p = line.split()
        if len(p) != 17:
            raise ValueError("Unexpected KITTI tracking columns")
        f, track = int(p[0]), int(p[1])
        cls = p[2]
        box = list(map(float, p[6:10]))
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Invalid reference box")
        # Custom mining reference, not official KITTI benchmark: >=25px visible car/person only.
        if cls in ("Car", "Pedestrian") and box[3] - box[1] >= 25 and int(p[4]) <= 2 and float(p[3]) <= 0.5:
            out.setdefault(f, []).append({"track": track, "label": 3 if cls == "Car" else 1, "box": box})
    return out


def missed(gold, preds):
    used = set()
    for p in sorted(preds, key=lambda x: -x["score"]):
        if p["score"] < 0.5:
            continue
        candidates = [
            (iou(g["box"], p["box"]), j)
            for j, g in enumerate(gold)
            if j not in used and g["label"] == p["label"]
        ]
        score, j = max(candidates, default=(0.0, -1))
        if score >= 0.5:
            used.add(j)
    return {g["track"] for j, g in enumerate(gold) if j not in used}


def clip_reference(clips, frames, labels):
    out = []
    for c in clips:
        tracks = set()
        counts = {}
        classes = set()
        for name in c["frames"]:
            f = int(Path(name).stem)
            gold = labels.get(f, [])
            tracks.update(g["track"] for g in gold)
            classes.update(g["label"] for g in gold)
            for t in missed(gold, frames[name]["predictions"]):
                counts[t] = counts.get(t, 0) + 1
        out.append(
            {
                "id": c["id"],
                "tracks": sorted(tracks),
                "classes": sorted(classes),
                "failure_tracks": sorted(t for t, n in counts.items() if n >= 2),
            }
        )
    return out


def evaluate(ref, chosen, sigs):
    allfail = set().union(*(set(r["failure_tracks"]) for r in ref))
    alltracks = set().union(*(set(r["tracks"]) for r in ref))
    failures = set().union(*(set(ref[i]["failure_tracks"]) for i in chosen))
    tracks = set().union(*(set(ref[i]["tracks"]) for i in chosen))
    x = np.asarray([sigs[i]["appearance"] for i in chosen])
    return dict(
        failure_coverage=len(failures) / len(allfail) if allfail else None,
        track_coverage=len(tracks) / len(alltracks) if alltracks else None,
        failure_hits=len(failures),
        failure_support=len(allfail),
        track_hits=len(tracks),
        track_support=len(alltracks),
        class_count=len(set().union(*(set(ref[i]["classes"]) for i in chosen))),
        redundant_track_fraction=1 - len(tracks) / sum(len(ref[i]["tracks"]) for i in chosen)
        if sum(len(ref[i]["tracks"]) for i in chosen)
        else None,
        appearance_cosine=float((x @ x.T)[np.triu_indices(len(chosen), 1)].mean()),
        clips=len(chosen),
        seconds=len(chosen),
    )
