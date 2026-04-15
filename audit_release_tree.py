#!/usr/bin/env python3
"""Audit the github/ subtree for accidental release bloat."""

from __future__ import annotations

import argparse
from pathlib import Path


BLOCKED_DIRS = {
    ".cache",
    "__pycache__",
    "logs",
    "outputs",
    "outputs_clean_rerun",
    "outputs_clean_rerun_test",
    "timeseries",
}

ALLOWED_LARGE_SUFFIXES = {
    "assets/gordon333MNI.nii.gz",
}


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def iter_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Path to the github/ release tree",
    )
    parser.add_argument(
        "--max-size-mb",
        default=5.0,
        type=float,
        help="Flag files larger than this threshold unless allowlisted",
    )
    parser.add_argument(
        "--top-n",
        default=25,
        type=int,
        help="Show the N largest files in the tree",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    size_limit = int(args.max_size_mb * 1024 * 1024)

    blocked_hits: list[Path] = []
    for name in BLOCKED_DIRS:
        candidate = root / name
        if candidate.exists():
            blocked_hits.append(candidate)

    all_files = list(iter_files(root))
    large_hits: list[Path] = []
    for path in all_files:
        rel = path.relative_to(root).as_posix()
        if rel in ALLOWED_LARGE_SUFFIXES:
            continue
        if path.stat().st_size > size_limit:
            large_hits.append(path)

    largest = sorted(all_files, key=lambda p: p.stat().st_size, reverse=True)[: args.top_n]

    print(f"Auditing release tree: {root}")
    print(f"Blocked directories present: {len(blocked_hits)}")
    for path in blocked_hits:
        print(f"  BLOCKED {path.relative_to(root)}")

    print(f"Oversize files (>{args.max_size_mb:.1f} MB): {len(large_hits)}")
    for path in sorted(large_hits):
        print(
            f"  LARGE {path.relative_to(root)} "
            f"({format_bytes(path.stat().st_size)})"
        )

    print(f"Largest {len(largest)} files:")
    for path in largest:
        print(
            f"  {format_bytes(path.stat().st_size):>8}  "
            f"{path.relative_to(root).as_posix()}"
        )

    if blocked_hits or large_hits:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
