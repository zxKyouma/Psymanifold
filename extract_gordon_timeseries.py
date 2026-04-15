#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from scipy.io import loadmat, savemat

from analysis_config import (
    DEFAULT_FD_SCRUB_THRESHOLD,
    DEFAULT_GORDON_CIFTI,
    DEFAULT_GORDON_VOL,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_MIN_RETAINED_FRAMES,
    DEFAULT_TIMESERIES_DIR,
    FD_SCRUB_DATASETS,
)


@dataclass(frozen=True)
class RowInfo:
    dataset: str
    subject_id: str
    session: str
    condition: str
    run_label: str
    input_type: str
    input_path: str
    confounds_path: str
    anatomy_path: str
    existing_timeseries_path: str
    mean_fd: str
    notes: str


PSICONNECT_INITIAL_FRAME_DROP = 5


def sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())


def row_output_path(root: Path, row: RowInfo, suffix: str = "") -> Path:
    fname = (
        f"{sanitize_token(row.subject_id)}_"
        f"{sanitize_token(row.session)}_"
        f"{sanitize_token(row.condition)}_"
        f"{sanitize_token(row.run_label)}_timeseries-Gordon333mean{suffix}.mat"
    )
    return root / row.dataset / sanitize_token(row.subject_id) / sanitize_token(row.session) / fname


def load_cached_volume_atlas(reference_bold: Path, atlas_path: Path, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(reference_bold))
    cache_key = "_".join(
        [
            "gordon333_vol",
            "x".join(str(v) for v in img.shape[:3]),
            "_".join(f"{v:.6f}" for v in img.affine.ravel()),
        ]
    )
    cache_path = cache_dir / f"{cache_key}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["labels"], cached["flat_atlas"]

    atlas_img = nib.load(str(atlas_path))
    target_3d = (img.shape[:3], img.affine)
    atlas_resampled = resample_from_to(atlas_img, target_3d, order=0)
    atlas = np.rint(np.asarray(atlas_resampled.dataobj)).astype(np.int16, copy=False)
    labels = np.unique(atlas)
    labels = labels[labels > 0]
    flat_atlas = atlas.reshape(-1)
    np.savez_compressed(cache_path, labels=labels, flat_atlas=flat_atlas)
    return labels, flat_atlas


def extract_volume_mean_timeseries(bold_path: Path, atlas_path: Path, cache_dir: Path) -> np.ndarray:
    labels, flat_atlas = load_cached_volume_atlas(bold_path, atlas_path, cache_dir)
    bold_img = nib.load(str(bold_path))
    bold = np.asarray(bold_img.dataobj, dtype=np.float32)
    if bold.ndim != 4:
        raise ValueError(f"Expected 4D BOLD image, got {bold.shape} for {bold_path}")
    flat_bold = bold.reshape(-1, bold.shape[-1])
    time_series = np.empty((bold.shape[-1], len(labels)), dtype=np.float32)
    for idx, label in enumerate(labels):
        roi_mask = flat_atlas == label
        if not roi_mask.any():
            raise ValueError(f"ROI {label} empty after resampling for {bold_path}")
        time_series[:, idx] = flat_bold[roi_mask].mean(axis=0, dtype=np.float32)
    return time_series


def maybe_drop_initial_frames(ts: np.ndarray, row: RowInfo) -> np.ndarray:
    if row.dataset == "PsiConnect" and row.input_type == "tedana_glm_bold":
        if ts.shape[0] <= PSICONNECT_INITIAL_FRAME_DROP:
            raise ValueError(
                f"PsiConnect run too short after initial frame drop: {ts.shape[0]} frames"
            )
        return ts[PSICONNECT_INITIAL_FRAME_DROP:, :]
    return ts


def extract_cifti_mean_timeseries(cifti_path: Path, dlabel_path: Path) -> np.ndarray:
    if not dlabel_path or not dlabel_path.exists():
        raise FileNotFoundError("A Gordon333 CIFTI dlabel path is required for scrubbed_dense_cifti inputs.")
    cimg = nib.load(str(cifti_path))
    dimg = nib.load(str(dlabel_path))
    cdata = cimg.get_fdata()
    timeseries = cdata.T.astype(np.float32, copy=False)
    labels = np.squeeze(dimg.get_fdata()).astype(np.int32, copy=False)
    if labels.shape[0] != timeseries.shape[0]:
        labels = labels[: timeseries.shape[0]]
    roi_labels = np.unique(labels)
    roi_labels = roi_labels[roi_labels > 0]
    out = np.empty((timeseries.shape[1], len(roi_labels)), dtype=np.float32)
    for idx, label in enumerate(roi_labels):
        roi_mask = labels == label
        out[:, idx] = timeseries[roi_mask].mean(axis=0, dtype=np.float32)
    return out


def extract_precomputed_roi_timeseries(mat_path: Path) -> np.ndarray:
    try:
        with h5py.File(mat_path, "r") as f:
            if "subject_data_for_adv_analysis" in f and "ts_subj" in f["subject_data_for_adv_analysis"]:
                ts = np.array(f["subject_data_for_adv_analysis"]["ts_subj"], dtype=np.float32)
                return ts
    except OSError:
        pass

    mat = loadmat(mat_path)
    if "BOLD_AAL" in mat:
        ts = np.asarray(mat["BOLD_AAL"], dtype=np.float32)
        if ts.ndim != 2:
            raise ValueError(f"Expected 2D BOLD_AAL in {mat_path}, got {ts.shape}")
        # Raw DMT files are ROI x time; convert to time x ROI.
        if ts.shape[0] < ts.shape[1]:
            return ts.T
        return ts

    raise ValueError(f"Unsupported precomputed ROI MAT format: {mat_path}")


def save_timeseries_mat(out_path: Path, time_series: np.ndarray, row: RowInfo, atlas_name: str, atlas_path: Path, extra: dict | None = None) -> None:
    payload = {
        "time_series": time_series.astype(np.float32, copy=False),
        "atlas_name": "Gordon333",
        "atlas_path": str(atlas_path),
        "roi_summary": "mean",
        "source_kind": row.input_type,
        "source_input": row.input_path,
        "dataset": row.dataset,
        "subject_id": row.subject_id,
        "session": row.session,
        "condition": row.condition,
        "run_label": row.run_label,
    }
    if extra:
        payload.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(out_path, payload, do_compression=True)


def infer_psiconnect_confounds(input_path: str) -> Path:
    return Path(input_path).parent / "confounds.mat"


def _match_length_fd(row: np.ndarray, n_out: int) -> np.ndarray | None:
    fd = np.asarray(row, dtype=float).reshape(-1)
    if len(fd) == n_out + 5:
        fd = fd[5:]
    if len(fd) != n_out:
        return None
    return fd


def load_psi_fd(confounds_path: Path, n_out: int, mean_fd_hint: float | None = None) -> tuple[np.ndarray, int]:
    with h5py.File(confounds_path, "r") as f:
        R = np.array(f["R"], dtype=float)

    candidates: list[tuple[int, np.ndarray, float]] = []
    for idx in range(R.shape[0]):
        fd = _match_length_fd(R[idx], n_out)
        if fd is None:
            continue
        if not np.isfinite(fd).any():
            continue
        # Plausible FD rows should be nonnegative and remain in a modest range.
        if np.nanmin(fd) < 0 or np.nanpercentile(fd, 99) > 10:
            continue
        mean_fd = float(np.nanmean(fd))
        candidates.append((idx, fd, mean_fd))

    if not candidates:
        raise ValueError(f"No plausible PsiConnect FD regressor found in {confounds_path}")

    if mean_fd_hint is not None and np.isfinite(mean_fd_hint):
        chosen = min(candidates, key=lambda item: abs(item[2] - float(mean_fd_hint)))
    else:
        chosen = min(candidates, key=lambda item: item[2])
    return chosen[1], int(chosen[0])


def load_lsd_fd(confounds_path: Path, n_out: int) -> np.ndarray:
    df = pd.read_csv(confounds_path, sep="\t")
    fd = pd.to_numeric(df["framewise_displacement"], errors="coerce").to_numpy(dtype=float)
    if len(fd) != n_out:
        raise ValueError(f"LSD FD length mismatch: {len(fd)} vs {n_out}")
    return fd


def scrub_time_series(ts: np.ndarray, row: RowInfo, fd_threshold: float) -> tuple[np.ndarray, int, float, float, int | None]:
    confounds_path = Path(row.confounds_path) if row.confounds_path else infer_psiconnect_confounds(row.input_path)
    mean_fd_pre = np.nan
    fd_row_index: int | None = None
    if row.dataset == "PsiConnect":
        mean_fd_hint = pd.to_numeric(pd.Series([row.mean_fd]), errors="coerce").iloc[0]
        fd, fd_row_index = load_psi_fd(confounds_path, ts.shape[0], mean_fd_hint=mean_fd_hint)
    elif row.dataset == "LSD":
        fd = load_lsd_fd(confounds_path, ts.shape[0])
    else:
        return ts, 0, np.nan, np.nan, None
    mean_fd_pre = float(np.nanmean(fd)) if np.isfinite(fd).any() else np.nan
    keep = np.isfinite(fd) & (fd <= fd_threshold)
    ts_scrubbed = ts[keep, :]
    mean_fd_post = float(np.nanmean(fd[keep])) if np.any(keep) else np.nan
    return ts_scrubbed, int(np.sum(~keep)), mean_fd_pre, mean_fd_post, fd_row_index


def build_raw_timeseries(row: RowInfo, gordon_vol: Path, gordon_cifti: Path, cache_dir: Path) -> np.ndarray:
    if row.input_type == "scrubbed_dense_cifti":
        ts = extract_cifti_mean_timeseries(Path(row.input_path), gordon_cifti)
        return maybe_drop_initial_frames(ts, row)
    if row.input_type in {"fmriprep_desc_preproc_bold", "tedana_glm_bold"}:
        ts = extract_volume_mean_timeseries(Path(row.input_path), gordon_vol, cache_dir)
        return maybe_drop_initial_frames(ts, row)
    if row.input_type == "precomputed_roi_timeseries_mat":
        ts = extract_precomputed_roi_timeseries(Path(row.input_path))
        return maybe_drop_initial_frames(ts, row)
    raise ValueError(f"Unsupported input_type={row.input_type}")


def process_row(row: RowInfo, out_root: Path, gordon_vol: Path, gordon_cifti: Path, cache_dir: Path, reuse_existing: bool, overwrite: bool, fd_threshold: float | None, min_retained_frames: int) -> dict:
    suffix = "_fdscrub" if fd_threshold is not None and row.dataset in FD_SCRUB_DATASETS else ""
    out_path: Path | None = row_output_path(out_root, row, suffix=suffix)
    status = "new"
    mean_fd_out = row.mean_fd
    retained_frames = np.nan
    scrubbed_frames = 0
    fd_row_index = np.nan

    if reuse_existing and row.existing_timeseries_path and (fd_threshold is None or row.dataset not in FD_SCRUB_DATASETS):
        out_path = Path(row.existing_timeseries_path)
        if row.dataset not in FD_SCRUB_DATASETS and "scrubbed" in row.input_type:
            status = "reused_existing_scrubbed_input"
        else:
            status = "reused_existing"
    elif out_path.exists() and not overwrite:
        status = "exists"
    else:
        time_series = build_raw_timeseries(row, gordon_vol, gordon_cifti, cache_dir)
        extra = {}
        if fd_threshold is not None and row.dataset in FD_SCRUB_DATASETS:
            time_series, n_scrub, mean_fd_pre, mean_fd_post, psi_fd_row = scrub_time_series(time_series, row, fd_threshold)
            retained_frames = int(time_series.shape[0])
            scrubbed_frames = int(n_scrub)
            mean_fd_out = mean_fd_post
            if psi_fd_row is not None:
                fd_row_index = int(psi_fd_row)
            extra["fd_scrub_threshold_mm"] = float(fd_threshold)
            extra["fd_scrubbed_frames"] = int(n_scrub)
            extra["mean_fd_pre_scrub"] = float(mean_fd_pre) if np.isfinite(mean_fd_pre) else np.nan
            extra["mean_fd_post_scrub"] = float(mean_fd_post) if np.isfinite(mean_fd_post) else np.nan
            if psi_fd_row is not None:
                extra["psi_fd_regressor_row"] = int(psi_fd_row)
            if retained_frames < min_retained_frames:
                status = "excluded_too_short_after_scrub"
                out_path = None
            else:
                status = "new_fd_scrub"
        elif row.dataset == "Psilocybin" and "scrubbed" in row.input_type:
            status = "new_from_scrubbed_input"
        if out_path is not None:
            save_timeseries_mat(out_path, time_series, row, "Gordon333", gordon_cifti if row.input_type == "scrubbed_dense_cifti" else gordon_vol, extra=extra)

    return {
        "dataset": row.dataset,
        "subject_id": row.subject_id,
        "session": row.session,
        "condition": row.condition,
        "run_label": row.run_label,
        "input_type": row.input_type,
        "input_path": row.input_path,
        "timeseries_path": str(out_path) if out_path is not None else "",
        "status": status,
        "mean_fd": mean_fd_out,
        "retained_frames": retained_frames,
        "scrubbed_frames": scrubbed_frames,
        "psi_fd_regressor_row": fd_row_index,
        "notes": row.notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build harmonized Gordon333 mean timeseries.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_DIR / "harmonized_manifest.csv")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--timeseries-root", type=Path, default=DEFAULT_TIMESERIES_DIR)
    parser.add_argument("--gordon-vol", type=Path, default=DEFAULT_GORDON_VOL)
    parser.add_argument("--gordon-cifti", type=Path, default=DEFAULT_GORDON_CIFTI)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / ".cache")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-reuse-existing", action="store_true")
    parser.add_argument("--scrub-fd-threshold", type=float, default=DEFAULT_FD_SCRUB_THRESHOLD)
    parser.add_argument("--min-retained-frames", type=int, default=DEFAULT_MIN_RETAINED_FRAMES)
    parser.add_argument("--no-scrub", action="store_true", help="Disable FD scrubbing and run the unscreened branch.")
    args = parser.parse_args()
    if args.no_scrub:
        args.scrub_fd_threshold = None

    df = pd.read_csv(args.manifest).fillna("")
    rows = [RowInfo(**rec) for rec in df.to_dict(orient="records")]
    records = []
    for idx, row in enumerate(rows, start=1):
        result = process_row(
            row=row,
            out_root=args.timeseries_root,
            gordon_vol=args.gordon_vol,
            gordon_cifti=args.gordon_cifti,
            cache_dir=args.cache_dir,
            reuse_existing=not args.no_reuse_existing,
            overwrite=args.overwrite,
            fd_threshold=args.scrub_fd_threshold,
            min_retained_frames=args.min_retained_frames,
        )
        records.append(result)
        print(f"[{idx}/{len(rows)}] {row.dataset} {row.subject_id} {row.session} -> {result['status']}")

    out_df = pd.DataFrame(records)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.outdir / "harmonized_timeseries_manifest.csv", index=False)
    out_df.to_csv(args.outdir / "harmonized_timeseries_manifest.tsv", index=False, sep="\t")
    summary = (
        out_df.groupby("dataset")
        .agg(
            n_rows=("dataset", "size"),
            n_subjects=("subject_id", "nunique"),
            n_conditions=("condition", "nunique"),
            new_files=("status", lambda s: int(s.isin(["new", "new_fd_scrub", "new_from_scrubbed_input"]).sum())),
            reused_existing=("status", lambda s: int((s == "reused_existing").sum())),
            existing=("status", lambda s: int((s == "exists").sum())),
        )
        .reset_index()
    )
    summary.to_csv(args.outdir / "harmonized_timeseries_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
