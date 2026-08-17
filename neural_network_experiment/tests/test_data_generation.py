from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import load_config
from src.data.synthetic_teacher_student import generate_replication


def test_fixed_teacher_reproducibility() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "experiments" / "smoke.yaml")
    rep_a = generate_replication(config, seed=7)
    rep_b = generate_replication(config, seed=7)
    assert rep_a.teacher_seed == rep_b.teacher_seed
    assert np.allclose(rep_a.train.x, rep_b.train.x)
    assert np.allclose(rep_a.train.y, rep_b.train.y)
    assert np.allclose(rep_a.test.x, rep_b.test.x)
    assert np.allclose(rep_a.test.y, rep_b.test.y)
    assert rep_a.train.x.shape[0] == config.data.n_total
    assert rep_a.test.x.shape[0] == config.data.n_total
    assert not np.allclose(rep_a.train.x, rep_a.test.x)
