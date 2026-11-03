import pytest

from ssam import build_learning_rate_policy, build_schedule


def test_piecewise_schedule():
    schedule = build_schedule({"name": "piecewise", "values": [[0, 1.0], [3, 0.1]]})
    assert schedule(2) == 1.0
    assert schedule(3) == 0.1


def test_piecewise_constant_schedule_has_explicit_plateaus():
    schedule = build_schedule(
        {
            "name": "piecewise_constant",
            "boundaries": [3, 7],
            "values": [1.0, 0.25, 0.05],
        }
    )

    assert [schedule(step) for step in (0, 2, 3, 6, 7, 100)] == [
        1.0,
        1.0,
        0.25,
        0.25,
        0.05,
        0.05,
    ]


@pytest.mark.parametrize(
    "config, message",
    [
        (
            {"name": "piecewise_constant", "boundaries": [3], "values": [1.0]},
            "one more value",
        ),
        (
            {
                "name": "piecewise_constant",
                "boundaries": [3, 3],
                "values": [1.0, 0.5, 0.1],
            },
            "strictly increasing",
        ),
        (
            {
                "name": "piecewise_constant",
                "boundaries": [3],
                "values": [1.0, -0.1],
            },
            "non-negative and finite",
        ),
    ],
)
def test_piecewise_constant_schedule_validates_plateaus(config, message):
    with pytest.raises(ValueError, match=message):
        build_schedule(config)


def test_unknown_schedule():
    with pytest.raises(ValueError, match="Unknown schedule"):
        build_schedule({"name": "unknown"})


def test_inverse_time_alpha_controls_decay_rate():
    slow = build_schedule(
        {"name": "inverse_time", "initial": 1.0, "alpha": 0.5}
    )
    fast = build_schedule(
        {"name": "inverse_time", "initial": 1.0, "alpha": 2.0}
    )

    assert slow(4) == pytest.approx(1.0 / 3.0)
    assert fast(4) == pytest.approx(1.0 / 9.0)


def test_tamed_rejects_vestigial_update_type():
    with pytest.raises(ValueError, match="Unknown tamed"):
        build_learning_rate_policy(
            {
                "name": "tamed",
                "type": "sgd",
                "inserted_lr": {"name": "constant", "value": 0.1},
            }
        )


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

