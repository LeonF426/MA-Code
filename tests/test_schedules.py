import pytest

from ssam import build_learning_rate_policy, build_schedule


def test_piecewise_schedule():
    schedule = build_schedule({"name": "piecewise", "values": [[0, 1.0], [3, 0.1]]})
    assert schedule(2) == 1.0
    assert schedule(3) == 0.1


def test_unknown_schedule():
    with pytest.raises(ValueError, match="Unknown schedule"):
        build_schedule({"name": "unknown"})


def test_static_learning_rate_policy_uses_uniform_interface():
    policy = build_learning_rate_policy({"name": "constant", "value": 0.2})
    assert policy(7, 3.0, 99.0) == 0.2
    assert not policy.requires_regularized_loss


def test_strong_descent_policy_uses_regularized_loss():
    policy = build_learning_rate_policy({
        "name": "strong_descent_diag",
        "dimension": 2,
        "depth": 1,
        "delta": 0.5,
        "safety": 0.5,
    })
    # C(2, 1) = 6, so safety * 2(1-delta)eta^2 / (C * loss).
    assert policy(0, 0.2, 0.5) == pytest.approx(0.5 * 0.04 / 3.0)
    with pytest.raises(ValueError, match="regularized-loss estimate"):
        policy(0, 0.2)

