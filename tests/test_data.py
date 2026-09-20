from types import SimpleNamespace

import pytest
import torch

from ssam import build_dataset, prepare_california_housing_data


def _arrays(samples: int = 20):
    features = torch.arange(samples * 8, dtype=torch.float32).reshape(samples, 8)
    targets = torch.linspace(0.5, 5.0, samples)
    return features, targets


def test_california_split_is_reproducible_and_disjoint():
    features, targets = _arrays()
    train = prepare_california_housing_data(
        features, targets, train=True, test_fraction=0.25, seed=4, standardize=False
    )
    test = prepare_california_housing_data(
        features, targets, train=False, test_fraction=0.25, seed=4, standardize=False
    )
    assert len(train) == 15
    assert len(test) == 5
    assert set(train.tensors[1].tolist()).isdisjoint(test.tensors[1].tolist())
    repeated = prepare_california_housing_data(
        features, targets, train=True, test_fraction=0.25, seed=4, standardize=False
    )
    torch.testing.assert_close(train.tensors[0], repeated.tensors[0])
    torch.testing.assert_close(train.tensors[1], repeated.tensors[1])


def test_california_standardization_is_fitted_on_training_partition():
    features, targets = _arrays()
    training = prepare_california_housing_data(
        features, targets, train=True, test_fraction=0.2, seed=3
    )
    torch.testing.assert_close(
        training.tensors[0].mean(dim=0),
        torch.zeros(8),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        training.tensors[0].std(dim=0, unbiased=False),
        torch.ones(8),
        atol=1e-6,
        rtol=0.0,
    )
    # Targets retain the dataset's original units by default ($100,000).
    assert 0.5 <= float(training.tensors[1].min())
    assert float(training.tensors[1].max()) <= 5.0


def test_california_cleaning_uses_only_training_statistics():
    features, targets = _arrays(20)
    permutation = torch.randperm(20, generator=torch.Generator().manual_seed(9))
    test_index = int(permutation[0])
    train_index = int(permutation[5])
    features[train_index, 2] = float("nan")
    features[test_index, 3] = 1e9

    training = prepare_california_housing_data(
        features,
        targets,
        train=True,
        test_fraction=0.2,
        seed=9,
        standardize=False,
        clip_quantiles=(0.1, 0.9),
    )
    test = prepare_california_housing_data(
        features,
        targets,
        train=False,
        test_fraction=0.2,
        seed=9,
        standardize=False,
        clip_quantiles=(0.1, 0.9),
    )

    assert torch.isfinite(training.tensors[0]).all()
    assert torch.isfinite(test.tensors[0]).all()
    # The held-out extreme value is clipped to a quantile learned without it.
    assert test.tensors[0][:, 3].max() <= training.tensors[0][:, 3].max()


def test_california_cleaning_can_reject_non_finite_values():
    features, targets = _arrays()
    features[0, 0] = float("inf")
    with pytest.raises(ValueError, match="enable impute_missing"):
        prepare_california_housing_data(
            features,
            targets,
            impute_missing=False,
        )

    features, targets = _arrays()
    targets[0] = float("nan")
    with pytest.raises(ValueError, match="targets must be finite"):
        prepare_california_housing_data(features, targets)


@pytest.mark.parametrize(
    "quantiles",
    [(0.9,), (-0.1, 0.9), (0.9, 0.1), (0.1, 1.1)],
)
def test_california_rejects_invalid_clip_quantiles(quantiles):
    features, targets = _arrays()
    with pytest.raises(ValueError, match="clip_quantiles"):
        prepare_california_housing_data(
            features,
            targets,
            clip_quantiles=quantiles,
        )


def test_build_dataset_fetches_california_housing_without_network(monkeypatch, tmp_path):
    datasets = pytest.importorskip("sklearn.datasets")
    features, targets = _arrays(10)
    observed = {}

    def fake_fetch(*, data_home, download_if_missing):
        observed.update(data_home=data_home, download=download_if_missing)
        return SimpleNamespace(data=features.numpy(), target=targets.numpy())

    monkeypatch.setattr(datasets, "fetch_california_housing", fake_fetch)
    dataset = build_dataset(
        {
            "name": "california_housing",
            "root": tmp_path,
            "test_size": 0.3,
            "standardize": False,
            "download": False,
            "seed": 2,
        },
        train=False,
    )
    assert len(dataset) == 3
    assert dataset.tensors[0].dtype == torch.float32
    assert observed == {"data_home": str(tmp_path), "download": False}

