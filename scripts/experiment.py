"""Frozen CPU inference, clip policy evaluation, clustered metric replay."""

import argparse
import importlib.metadata
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.transforms.functional import pil_to_tensor

from miner import (
    check_clips,
    choose,
    clip_reference,
    digest,
    evaluate,
    file_hash,
    immutable,
    read,
    reference,
    signatures,
)

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models/detector.pth"
SEEDS = [101, 211, 307, 409, 503]
METHODS = ["random", "uniform", "dedup", "appearance_facility", "temporal"]


def identity():
    return dict(
        manifest=file_hash(ROOT / "reports/manifest.json"),
        weights=file_hash(WEIGHTS),
        sources={
            str(p.relative_to(ROOT)): file_hash(p)
            for p in sorted(list((ROOT / "src").glob("*.py")) + list((ROOT / "scripts").glob("*.py")))
        },
        runner=file_hash(Path(__file__)),
        lock=file_hash(ROOT / "uv.lock"),
        versions={n: importlib.metadata.version(n) for n in ("torch", "torchvision", "numpy", "Pillow")},
    )


def freeze():
    m = read(ROOT / "reports/manifest.json")
    check_clips(m["clips"])
    immutable(
        ROOT / "reports/protocol.json",
        dict(
            identity=identity(),
            seeds=SEEDS,
            methods=METHODS,
            budget_clips=2,
            budget_seconds=2,
            model="torchvision fasterrcnn_mobilenet_v3_large_320_fpn COCO_V1; CPU float32 4 threads; no training",
            input="RGB float32 [3,H,W], shorter side320/max640, score retention .05; 5Hz sampled observations of one-second 10Hz source clips",
            primary="Mean per-heldout-sequence distinct missed-reference-track coverage temporal minus random; mean paired seeds then 2000 sequence bootstrap, seed 911205",
            reference="Car and Pedestrian; box height>=25px, truncation<=.5, occlusion<=2; score>=.5 same-class IoU>=.5 one-to-one; failure track missed in >=2 observed frames in a clip; custom mining endpoint, NOT official KITTI AP/HOTA",
            method="Greedy facility location on equal-weight concatenation of normalized mean RGB histograms and normalized mean absolute temporal histogram differences; no label input",
            statistics="Evaluation is applying a frozen mining policy to unseen sequence pixels; reference labels read ONLY after all selections saved. No model selection, tuning or training on eval sequences. Pool and dev remain separate auxiliary policy checks; do not mix into primary.",
            stopping="One frozen comparison, all failures retained; no search for positive result; zero-support sequence coverage stays null with explicit support, other metrics retained",
            boundaries="20 source sequences, 6 disjoint clips each; two selected seconds per sequence; 5Hz observation misses faster events; KITTI small daytime scope; no proof of downstream training benefit",
        ),
    )


def validate():
    p = read(ROOT / "reports/protocol.json")
    if p["identity"] != identity():
        raise ValueError("Frozen protocol drift")
    m = read(ROOT / "reports/manifest.json")
    check_clips(m["clips"])
    for r in m["frames"]:
        if file_hash(ROOT / "data" / r["id"]) != r["sha256"]:
            raise ValueError("Frame drift")
    return p, m


def run(limit=None):
    p, m = validate()
    torch.set_num_threads(4)
    torch.manual_seed(911205)
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None, box_score_thresh=0.05)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.eval()
    done = 0
    hits = 0
    started = time.perf_counter()
    for r in m["frames"]:
        path = ROOT / "reports/frames" / f"{r['sequence']}-{r['frame']:06d}.json"
        key = digest([digest(p), r["id"], r["sha256"]])
        if path.exists():
            cached = read(path)
            if cached["key"] != key or cached.get("status") != "success":
                raise ValueError("Bad cache")
            hits += 1
            continue
        if limit is not None and done >= limit:
            break
        t = time.perf_counter()
        try:
            with Image.open(ROOT / "data" / r["id"]) as im:
                im = im.convert("RGB")
                a = np.asarray(im.resize((64, 32)))
                hist = np.concatenate(
                    [np.histogram(a[:, :, c], bins=8, range=(0, 256))[0] for c in range(3)]
                ).astype(float)
                hist /= hist.sum()
                x = pil_to_tensor(im).float() / 255
            with torch.inference_mode():
                pred = model([x])[0]
            boxes = pred["boxes"].tolist()
            scores = pred["scores"].tolist()
            labels = pred["labels"].tolist()
            raw = [
                dict(box=b, score=s, label=label)
                for b, s, label in zip(boxes, scores, labels)
                if label in (1, 3)
            ]
            immutable(
                path,
                dict(
                    key=key,
                    status="success",
                    id=r["id"],
                    sha256=r["sha256"],
                    appearance=hist.tolist(),
                    predictions=raw,
                    seconds=time.perf_counter() - t,
                ),
            )
        except Exception as e:
            immutable(
                ROOT / "reports/errors" / f"{r['sequence']}-{r['frame']:06d}-{time.time_ns()}.json",
                dict(key=key, error=type(e).__name__, message=str(e)),
            )
            raise
        done += 1
        if done % 100 == 0:
            print(f"inferred {done} cache {hits}", flush=True)
    print(dict(computed=done, cache_hits=hits, seconds=time.perf_counter() - started))


def load_frames(p, m):
    frames = {}
    for r in m["frames"]:
        q = read(ROOT / "reports/frames" / f"{r['sequence']}-{r['frame']:06d}.json")
        if q["key"] != digest([digest(p), r["id"], r["sha256"]]) or q["status"] != "success":
            raise ValueError("Raw identity mismatch")
        frames[r["id"]] = q
    return frames


def analyze(verify=False):
    p, m = validate()
    frames = load_frames(p, m)
    plans = []
    byseq = {}
    sigs = {}
    for seq in m["splits"]:
        clips = [c for c in m["clips"] if c["sequence"] == seq]
        byseq[seq] = clips
        sigs[seq] = signatures(clips, frames)
        for method in METHODS:
            for seed in SEEDS:
                plans.append(
                    dict(
                        sequence=seq,
                        split=m["splits"][seq],
                        method=method,
                        seed=seed,
                        chosen=choose(sigs[seq], method, seed),
                    )
                )
    # Durable boundary: selections are immutable before opening reference labels.
    immutable(ROOT / "reports/selections.json", dict(protocol=digest(p), plans=plans))
    refs = {}
    for seq in m["splits"]:
        label = ROOT / f"data/labels/{seq}.txt"
        if file_hash(label) != m["label_hashes"][seq]:
            raise ValueError("Reference drift")
        refs[seq] = clip_reference(byseq[seq], frames, reference(label.read_text()))
    results = [
        {**plan, **evaluate(refs[plan["sequence"]], plan["chosen"], sigs[plan["sequence"]])} for plan in plans
    ]
    paired = []
    for seq, split in m["splits"].items():
        if split != "eval":
            continue
        rr = [r for r in results if r["sequence"] == seq]
        values = {method: [r["failure_coverage"] for r in rr if r["method"] == method] for method in METHODS}
        if any(x is None for x in values["random"]):
            paired.append(dict(sequence=seq, delta=None))
            continue
        paired.append(
            dict(sequence=seq, delta=float(np.mean(values["temporal"]) - np.mean(values["random"])))
        )
    vals = [r["delta"] for r in paired if r["delta"] is not None]
    rng = np.random.default_rng(911205)
    boot = np.mean(rng.choice(vals, size=(2000, len(vals)), replace=True), axis=1) if vals else []
    summary = dict(
        primary_delta=float(np.mean(vals)) if vals else None,
        ci95=np.quantile(boot, [0.025, 0.975]).tolist() if vals else None,
        paired_sequences=paired,
        evaluation_sequences=sum(v == "eval" for v in m["splits"].values()),
        supported_sequences=len(vals),
        rows=results,
        references=refs,
        frame_count=len(frames),
        compute_seconds=sum(f["seconds"] for f in frames.values()),
        decision="No downstream training claim. Positive mining claim requires primary interval lower bound above zero; otherwise retain negative/inconclusive result.",
    )
    if verify:
        if summary != read(ROOT / "reports/result.json"):
            raise ValueError("Metric replay mismatch")
        print("Verified all selections, references, budgets and sequence bootstrap from raw evidence")
    else:
        immutable(ROOT / "reports/result.json", summary)
        print({k: v for k, v in summary.items() if k not in ("rows", "references")})


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("action", choices=["freeze", "run", "analyze", "verify"])
    a.add_argument("--limit", type=int)
    args = a.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "run":
        run(args.limit)
    else:
        analyze(args.action == "verify")
