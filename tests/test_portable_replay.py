import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("portable", Path(__file__).parents[1] / "portable_replay.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def test_witness_rejects_suboptimal_and_bad_budget():
    items = [dict(id=str(i), appearance=x, temporal=x) for i, x in enumerate([[1, 0], [1, 0], [0, 1]])]
    p.validate_witness(items, "temporal", 101, [0, 2])
    with pytest.raises(ValueError, match="maximizer"):
        p.validate_witness(items, "temporal", 101, [0, 1])
    with pytest.raises(ValueError):
        p.validate_witness(items, "random", 101, [0, 0])
