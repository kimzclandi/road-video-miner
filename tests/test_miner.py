import copy

import numpy as np
import pytest

from miner import check_clips, choose, evaluate, immutable, missed, reference, signatures


def clip(start=0, seq="0000", split="eval"):
    return dict(
        id=f"{seq}-{start}",
        sequence=seq,
        split=split,
        start_frame=start,
        end_frame_exclusive=start + 10,
        fps=10,
        seconds=1,
        frames=[f"training/image_02/{seq}/{i:06d}.png" for i in range(start, start + 10, 2)],
    )


def test_temporal_contract():
    check_clips([clip(), clip(10)])
    for bad in ([clip(), clip(8)], [clip(), clip(10, split="dev")], [clip(), clip()]):
        with pytest.raises(ValueError):
            check_clips(bad)
    c = clip()
    c["frames"][1] = c["frames"][0]
    with pytest.raises(ValueError):
        check_clips([c])
    c = clip()
    c["frames"][1] = "training/image_02/9999/000002.png"
    with pytest.raises(ValueError):
        check_clips([c])


def test_selector_no_labels_and_equal_budget():
    items = [
        dict(id=str(i), appearance=[float(i == 0), float(i != 0)], temporal=[float(i == 0), float(i != 0)])
        for i in range(6)
    ]
    for method in ("random", "uniform", "dedup", "temporal"):
        selected = choose(items, method, 101)
        assert len(set(selected)) == 2
        assert selected == choose(items, method, 101)
    bad = copy.deepcopy(items)
    bad[0]["failure_tracks"] = [1]
    with pytest.raises(ValueError, match="Unlabeled"):
        choose(bad, "temporal", 1)


def test_temporal_distinguishes_order_change():
    c = clip()
    f = {name: {"appearance": ([1, 0] if i % 2 else [0, 1])} for i, name in enumerate(c["frames"])}
    s = signatures([c], f)[0]
    assert np.isfinite(s["temporal"]).all()
    assert len(s["temporal"]) == 4


def test_reference_matching_keeps_misses_and_one_to_one():
    g = [dict(track=1, label=3, box=[0, 0, 30, 30]), dict(track=2, label=3, box=[0, 0, 30, 30])]
    assert missed(g, []) == {1, 2}
    assert len(missed(g, [dict(label=3, box=[0, 0, 30, 30], score=0.9)])) == 1
    assert missed(g, [dict(label=1, box=[0, 0, 30, 30], score=0.9)]) == {1, 2}
    ref = reference("0 1 Car 0 0 0 0 0 30 30 1 1 1 1 1 1 1")
    assert ref[0][0]["track"] == 1
    with pytest.raises(ValueError):
        reference("broken")


def test_no_support_is_not_perfect_score(tmp_path):
    refs = [dict(tracks=[], failure_tracks=[], classes=[]) for _ in range(2)]
    sigs = [dict(appearance=[1, 0]) for _ in range(2)]
    r = evaluate(refs, [0, 1], sigs)
    assert r["failure_coverage"] is None and r["seconds"] == 2
    p = tmp_path / "e.json"
    immutable(p, r)
    immutable(p, r)
    with pytest.raises(ValueError):
        immutable(p, {"different": True})
