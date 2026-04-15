from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def output_path(*parts: str) -> Path:
    return repo_path("outputs", *parts)


def env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    return Path(raw).expanduser()


def configured_path(name: str, default_rel: str | None = None) -> Path | None:
    path = env_path(name)
    if path is not None:
        return path
    if default_rel is None:
        return None
    return repo_path(*Path(default_rel).parts)


def require_path(name: str, description: str, default_rel: str | None = None) -> Path:
    path = configured_path(name, default_rel=default_rel)
    if path is None:
        raise FileNotFoundError(
            f"Missing required input for {description}. Set environment variable {name}"
            + (f" or provide the repo-relative default at {default_rel}." if default_rel else ".")
        )
    if not path.exists():
        raise FileNotFoundError(
            f"Configured path for {description} does not exist: {path} (from {name})"
        )
    return path


def optional_existing_path(name: str, default_rel: str | None = None) -> Path | None:
    path = configured_path(name, default_rel=default_rel)
    if path is None or not path.exists():
        return None
    return path
