"""Offline public evidence replay; full reference-label replay uses experiment.py verify."""

from pathlib import Path

from miner import check_clips, choose, digest, evaluate, file_hash, read, signatures

root = Path(__file__).resolve().parents[1]
p = read(root / "reports/protocol.json")
m = read(root / "reports/manifest.json")
result = read(root / "reports/result.json")
assert file_hash(root / "reports/manifest.json") == p["identity"]["manifest"]
for path, sha in p["identity"]["sources"].items():
    assert file_hash(root / path) == sha, path
check_clips(m["clips"])
frames = {}
for r in m["frames"]:
    q = read(root / "reports/frames" / f"{r['sequence']}-{r['frame']:06d}.json")
    assert q["key"] == digest([digest(p), r["id"], r["sha256"]]) and q["status"] == "success"
    frames[r["id"]] = q
plans = read(root / "reports/selections.json")
assert plans["protocol"] == digest(p)
rows = []
for plan in plans["plans"]:
    clips = [c for c in m["clips"] if c["sequence"] == plan["sequence"]]
    sig = signatures(clips, frames)
    assert choose(sig, plan["method"], plan["seed"]) == plan["chosen"]
    rows.append({**plan, **evaluate(result["references"][plan["sequence"]], plan["chosen"], sig)})
assert rows == result["rows"]
print(
    f"Verified {len(frames)} raw frames, {len(rows)} policy evaluations and fixed budgets; reference extraction independently checked locally"
)
