import torch

from ssam.pinn_batching import PINNBatcher


def test_matched_sgd_and_ssam_receive_the_same_batches():
    groups = {
        "interior": torch.arange(16).reshape(8, 2),
        "boundary": torch.arange(8).reshape(4, 2),
    }
    base = {"batch_size": 4, "batch_seed": 19, "seed": 3}
    sgd = PINNBatcher(groups, {**base, "algorithm": "sgd"})
    ssam = PINNBatcher(groups, {**base, "algorithm": "s_sam"})

    for _ in range(4):
        sgd_batch = sgd.next()
        ssam_batch = ssam.next()
        for name in groups:
            torch.testing.assert_close(sgd_batch[name], ssam_batch[name])


def test_batch_seed_changes_the_collocation_sequence():
    groups = {"interior": torch.arange(16).reshape(8, 2)}
    first = PINNBatcher(
        groups,
        {"algorithm": "sgd", "batch_size": 4, "batch_seed": 1},
    ).next()
    second = PINNBatcher(
        groups,
        {"algorithm": "sgd", "batch_size": 4, "batch_seed": 2},
    ).next()

    assert not torch.equal(first["interior"], second["interior"])

