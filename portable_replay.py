"""Validate recorded greedy decisions as witnesses; do not pretend cross-BLAS tie identity."""

import random
from pathlib import Path

import numpy as np

from miner import check_clips, choose, digest, evaluate, file_hash, read, signatures


def validate_witness(items, method, seed, chosen):
    if len(chosen) != 2 or len(set(chosen)) != 2 or any(i < 0 or i >= len(items) for i in chosen):
        raise ValueError("Budget/identity mismatch")
    if method in ("random", "uniform"):
        if chosen != choose(items, method, seed):
            raise ValueError("Discrete policy mismatch")
        return
    x = np.asarray([r["temporal" if method == "temporal" else "appearance"] for r in items])
    sim = np.clip(x @ x.T, 0, 1)
    coverage = np.zeros(len(items))
    selected = []
    tie = list(range(len(items)))
    random.Random(seed).shuffle(tie)
    for index in chosen:
        candidates = [i for i in tie if i not in selected]
        gains = {
            j: (1.0 - max((sim[j, k] for k in selected), default=0.0))
            if method == "dedup"
            else float(np.maximum(coverage, sim[:, j]).sum() - coverage.sum())
            for j in candidates
        }
        if max(gains.values()) - gains[index] > 1e-12:
            raise ValueError("Recorded selection is not a greedy maximizer")
        selected.append(index)
        coverage = np.maximum(coverage, sim[:, index])


def main():
    root = Path(__file__).resolve().parent
    p = read(root / "reports/protocol.json")
    m = read(root / "reports/manifest.json")
    r = read(root / "reports/result.json")
    if file_hash(root / "reports/manifest.json") != p["identity"]["manifest"]:
        raise ValueError("Manifest drift")
    for path, sha in p["identity"]["sources"].items():
        if file_hash(root / path) != sha:
            raise ValueError("Frozen source drift")
    check_clips(m["clips"])
    frames = {}
    for f in m["frames"]:
        q = read(root / "reports/frames" / f"{f['sequence']}-{f['frame']:06d}.json")
        if q["key"] != digest([digest(p), f["id"], f["sha256"]]) or q["status"] != "success":
            raise ValueError("Raw record identity")
        frames[f["id"]] = q
    plans = read(root / "reports/selections.json")
    different = []
    computed = []
    if plans["protocol"] != digest(p):
        raise ValueError("Selection identity")
    for plan in plans["plans"]:
        sig = signatures([c for c in m["clips"] if c["sequence"] == plan["sequence"]], frames)
        validate_witness(sig, plan["method"], plan["seed"], plan["chosen"])
        regenerated = choose(sig, plan["method"], plan["seed"])
        if regenerated != plan["chosen"]:
            different.append({**plan, "regenerated": regenerated})
        computed.append({**plan, **evaluate(r["references"][plan["sequence"]], plan["chosen"], sig)})
    for a, b in zip(computed, r["rows"], strict=True):
        for key in a:
            if isinstance(a[key], float):
                if not np.isclose(a[key], b[key], rtol=0, atol=1e-12):
                    raise ValueError("Metric drift")
            elif a[key] != b[key]:
                raise ValueError("Discrete evidence mismatch")
    vals = []
    for seq in [s for s, split in m["splits"].items() if split == "eval"]:
        rr = [x for x in computed if x["sequence"] == seq]
        if rr[0]["failure_coverage"] is None:
            continue
        vals.append(
            np.mean([x["failure_coverage"] for x in rr if x["method"] == "temporal"])
            - np.mean([x["failure_coverage"] for x in rr if x["method"] == "random"])
        )
    ci = np.quantile(
        np.mean(np.random.default_rng(911205).choice(vals, size=(2000, len(vals))), axis=1), [0.025, 0.975]
    )
    if not np.allclose(ci, r["ci95"], rtol=0, atol=1e-12) or not np.isclose(
        np.mean(vals), r["primary_delta"], rtol=0, atol=1e-12
    ):
        raise ValueError("Bootstrap mismatch")
    print(
        f"Validated {len(computed)} recorded decisions, original metrics and bootstrap; {len(different)} cross-platform near-tie regenerations differ."
    )
    print(different[:10])


if __name__ == "__main__":
    main()
