from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing import get_type_hints

import yaml

from src.utils import deep_update


@dataclass
class ExperimentConfig:
    name: str = "robust_cv_regression"
    output_root: str = "outputs"
    run_name: str | None = None
    seed: int = 20260408
    device: str = "auto"
    parallel_workers: int = 1
    save_checkpoints: bool = True
    log_every_epoch: bool = True


@dataclass
class DataConfig:
    d_core: int = 4
    d_frag: int = 4
    d_noise: int = 8
    standardize_blocks: bool = True
    regenerate_teacher_each_replication: bool = False
    n_total: int = 1000


@dataclass
class TeacherConfig:
    m_core: int = 16
    m_frag: int = 16
    sigma: float = 0.25
    lambda_value: float = 1.0
    teacher_seed: int = 1234
    data_seed_offset: int = 100000
    test_seed_offset: int = 200000


@dataclass
class ModelConfig:
    width: int = 64
    depth: int = 3
    activation: str = "relu"
    dropout: float = 0.0


@dataclass
class AttackConfig:
    family: str = "linf"
    radius: float = 0.1
    steps: int = 7
    step_size: float = 0.03
    restarts: int = 1
    random_start: bool = True


@dataclass
class StochasticRobustConfig:
    num_samples: int = 8
    distribution: str = "uniform"


@dataclass
class AttacksConfig:
    train: AttackConfig = field(default_factory=AttackConfig)
    eval: AttackConfig = field(default_factory=lambda: AttackConfig(radius=0.2, steps=20, restarts=4))
    sr: StochasticRobustConfig = field(default_factory=StochasticRobustConfig)


@dataclass
class TrainingConfig:
    epochs: int = 80
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 12
    early_stopping_metric: str = "stochastic"
    eval_on_train_end: bool = True


@dataclass
class CVConfig:
    method: str = "repeated_holdout"
    holdout_ratio: float = 0.8
    repeats: int = 5
    num_folds: int = 5
    tie_break_order: list[str] = field(default_factory=lambda: ["core_only", "sr", "at", "erm"])


@dataclass
class GridConfig:
    procedures: list[str] = field(default_factory=lambda: ["erm", "at", "sr", "core_only"])
    n_total: list[int] = field(default_factory=lambda: [1000, 2000])
    lambda_values: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
    eval_radii: list[float] = field(default_factory=lambda: [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2])
    train_radii: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    threat_models: list[str] = field(default_factory=lambda: ["linf", "l2"])
    num_replications: int = 50
    kfold_enabled: bool = False


@dataclass
class PlotConfig:
    dpi: int = 180
    file_formats: list[str] = field(default_factory=lambda: ["png", "pdf"])


@dataclass
class RootConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    attacks: AttacksConfig = field(default_factory=AttacksConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")
    return data


def load_raw_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    raw = _read_yaml(path)
    extends = raw.pop("extends", None)
    if extends is None:
        return raw
    base = load_raw_config(path.parent / extends)
    return deep_update(base, raw)


def _build_dataclass(cls: type[Any], payload: dict[str, Any]) -> Any:
    type_hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for field_info in cls.__dataclass_fields__.values():  # type: ignore[attr-defined]
        if field_info.name not in payload:
            continue
        field_value = payload[field_info.name]
        field_type = type_hints.get(field_info.name, field_info.type)
        if hasattr(field_type, "__dataclass_fields__") and isinstance(field_value, dict):
            values[field_info.name] = _build_dataclass(field_type, field_value)
        else:
            values[field_info.name] = field_value
    return cls(**values)


def load_config(config_path: str | Path) -> RootConfig:
    payload = load_raw_config(config_path)
    return RootConfig(
        experiment=_build_dataclass(ExperimentConfig, payload.get("experiment", {})),
        data=_build_dataclass(DataConfig, payload.get("data", {})),
        teacher=_build_dataclass(TeacherConfig, payload.get("teacher", {})),
        model=_build_dataclass(ModelConfig, payload.get("model", {})),
        training=_build_dataclass(TrainingConfig, payload.get("training", {})),
        attacks=AttacksConfig(
            train=_build_dataclass(AttackConfig, payload.get("attacks", {}).get("train", {})),
            eval=_build_dataclass(AttackConfig, payload.get("attacks", {}).get("eval", {})),
            sr=_build_dataclass(StochasticRobustConfig, payload.get("attacks", {}).get("sr", {})),
        ),
        cv=_build_dataclass(CVConfig, payload.get("cv", {})),
        grid=_build_dataclass(GridConfig, payload.get("grid", {})),
        plots=_build_dataclass(PlotConfig, payload.get("plots", {})),
    )
