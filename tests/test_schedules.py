import pytest

from ssam import build_schedule


def test_piecewise_schedule():
    schedule = build_schedule({"name": "piecewise", "values": [[0, 1.0], [3, 0.1]]})
    assert schedule(2) == 1.0
    assert schedule(3) == 0.1


def test_unknown_schedule():
    with pytest.raises(ValueError, match="Unknown schedule"):
        build_schedule({"name": "unknown"})
