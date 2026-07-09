from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path)
    return config


def project_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_data_file(config: dict[str, Any]) -> Path:
    data = config["data"]
    path = project_path(data["path"])
    if not path.exists():
        raise FileNotFoundError(f"Missing CCPP data file: {path}")
    actual = sha256_file(path)
    expected = str(data["sha256"]).upper()
    if actual != expected:
        raise ValueError(
            f"Data checksum mismatch for {path}: expected {expected}, got {actual}"
        )
    return path
