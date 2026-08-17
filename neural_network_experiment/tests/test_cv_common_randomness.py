from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.config import load_config  # noqa: E402
from src.cv.selection import make_common_randomness, make_disjoint_holdout_splits  # noqa: E402
from src.data.synthetic_teacher_student import generate_replication  # noqa: E402


def test_common_randomness_is_reproducible() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "experiments" / "smoke.yaml")
    replication = generate_replication(config, seed=11)
    noise_a = make_common_randomness(replication.train, config.attacks.eval, num_samples=3, seed=99)
    noise_b = make_common_randomness(replication.train, config.attacks.eval, num_samples=3, seed=99)
    assert np.allclose(noise_a.numpy(), noise_b.numpy())


def test_disjoint_holdout_splits_train_once_and_validate_on_complement() -> None:
    n_samples = 103
    splits = make_disjoint_holdout_splits(n_samples=n_samples, repeats=5, seed=123)
    assert len(splits) == 5

    all_indices = set(range(n_samples))
    train_sets = [set(train_idx.tolist()) for train_idx, _ in splits]
    assert set.union(*train_sets) == all_indices
    for left in range(len(train_sets)):
        for right in range(left + 1, len(train_sets)):
            assert train_sets[left].isdisjoint(train_sets[right])

    for train_idx, val_idx in splits:
        train_set = set(train_idx.tolist())
        val_set = set(val_idx.tolist())
        assert train_set.isdisjoint(val_set)
        assert val_set == all_indices - train_set
        assert len(train_idx) in {n_samples // 5, n_samples // 5 + 1}
