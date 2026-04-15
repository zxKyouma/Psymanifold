#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from analysis_config import DEFAULT_FD_SCRUB_THRESHOLD, DEFAULT_MANIFEST_DIR, DEFAULT_MIN_RETAINED_FRAMES, FD_SCRUB_DATASETS
from extract_gordon_timeseries import RowInfo, infer_psiconnect_confounds, maybe_drop_initial_frames, scrub_time_series


DEFAULT_OPTIONAL_OUTDIR = Path(__file__).resolve().parent / "outputs" / "harmonized_psychedelics_optional"
SUPPORTED_DATASETS = {"PsiConnect", "Psilocybin"}


def load_highres_matrix(row: RowInfo) -> np.ndarray:
    p = Path(row.input_path)
    if row.input_type == "scrubbed_dense_cifti":
        cimg = nib.load(str(p))
        data = np.asarray(cimg.get_fdata(), dtype=np.float32)
        return maybe_drop_initial_frames(data.T, row)
    if row.input_type in {"fmriprep_desc_preproc_bold", "tedana_glm_bold"}:
        img = nib.load(str(p))
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim != 4:
            raise ValueError(f"Expected 4D input for {p}, got {data.shape}")
        t = data.shape[-1]
        flat = data.reshape(-1, t)
        finite_mask = np.all(np.isfinite(flat), axis=1)
        var_mask = np.nanstd(flat, axis=1) > 1e-8
        keep = finite_mask & var_mask
        if not np.any(keep):
            raise ValueError(f"No valid nonzero-variance high-resolution features in {p}")
        return maybe_drop_initial_frames(flat[keep].T, row)
    raise ValueError(f"Unsupported input_type for high-resolution complexity: {row.input_type}")


def compute_ngsc_highres(ts_txv: np.ndarray) -> float:
    if ts_txv.ndim != 2 or min(ts_txv.shape) < 2:
        return np.nan
    y = np.asarray(ts_txv, dtype=np.float32, order="C")
    y = y - np.mean(y, axis=0, keepdims=True, dtype=np.float32)
    sd = np.std(y, axis=0, ddof=0, keepdims=True, dtype=np.float32)
    sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
    y = y / sd
    y[~np.isfinite(y)] = 0.0
    gram = (y @ y.T).astype(np.float64, copy=False) / max(y.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(gram)
    eigvals = eigvals[np.isfinite(eigvals) & (eigvals > 1e-9)]
    if eigvals.size < 2:
        return 0.0
    p = eigvals / eigvals.sum()
    h = -np.sum(p * np.log2(p))
    return float(h / np.log2(len(eigvals)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute optional high-resolution NGSC_vtx for supported datasets.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_DIR / "harmonized_manifest.csv")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OPTIONAL_OUTDIR)
    parser.add_argument("--datasets", nargs="*", default=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--scrub-fd-threshold", type=float, default=DEFAULT_FD_SCRUB_THRESHOLD)
    parser.add_argument("--min-retained-frames", type=int, default=DEFAULT_MIN_RETAINED_FRAMES)
    parser.add_argument("--no-scrub", action="store_true", help="Disable FD scrubbing for high-resolution complexity.")
    args = parser.parse_args()
    if args.no_scrub:
        args.scrub_fd_threshold = None

    wanted = set(args.datasets)
    df = pd.read_csv(args.manifest).fillna("")
    records = []
    for rec in df.to_dict(orient="records"):
        row = RowInfo(**rec)
        if row.dataset not in wanted or row.dataset not in SUPPORTED_DATASETS:
            continue
        try:
            print(f"{row.dataset} {row.subject_id} {row.session}: loading", flush=True)
            ts = load_highres_matrix(row)
            scrubbed_frames = 0
            retained_frames = ts.shape[0]
            mean_fd_out = pd.to_numeric(pd.Series([row.mean_fd]), errors="coerce").iloc[0]
            psi_fd_row = np.nan
            if args.scrub_fd_threshold is not None and row.dataset in FD_SCRUB_DATASETS:
                ts, scrubbed_frames, _, mean_fd_post, psi_fd_row = scrub_time_series(ts, row, args.scrub_fd_threshold)
                retained_frames = ts.shape[0]
                mean_fd_out = mean_fd_post
                if retained_frames < args.min_retained_frames:
                    records.append({
                        "Dataset": row.dataset,
                        "SubjectID": row.subject_id,
                        "Session": row.session,
                        "Condition": row.condition,
                        "RunLabel": row.run_label,
                        "FileKey": f"{row.dataset}|{row.subject_id}|{row.session}|{row.run_label}",
                        "MeanFD": mean_fd_out,
                        "NGSC_vtx": np.nan,
                        "NHighResFeatures": np.nan,
                        "NumFrames": retained_frames,
                        "ScrubbedFrames": scrubbed_frames,
                        "PsiFDRegressorRow": psi_fd_row,
                        "Status": "excluded_too_short_after_scrub",
                    })
                    print(f"{row.dataset} {row.subject_id} {row.session}: excluded_too_short_after_scrub ({retained_frames} frames)", flush=True)
                    continue
            ngsc_vtx = compute_ngsc_highres(ts)
            records.append({
                "Dataset": row.dataset,
                "SubjectID": row.subject_id,
                "Session": row.session,
                "Condition": row.condition,
                "RunLabel": row.run_label,
                "FileKey": f"{row.dataset}|{row.subject_id}|{row.session}|{row.run_label}",
                "MeanFD": mean_fd_out,
                "NGSC_vtx": ngsc_vtx,
                "NHighResFeatures": ts.shape[1],
                "NumFrames": retained_frames,
                "ScrubbedFrames": scrubbed_frames,
                "PsiFDRegressorRow": psi_fd_row,
                "Status": "ok",
            })
            print(f"{row.dataset} {row.subject_id} {row.session}: ok ({ts.shape[0]} x {ts.shape[1]})", flush=True)
        except Exception as exc:
            records.append({
                "Dataset": row.dataset,
                "SubjectID": row.subject_id,
                "Session": row.session,
                "Condition": row.condition,
                "RunLabel": row.run_label,
                "FileKey": f"{row.dataset}|{row.subject_id}|{row.session}|{row.run_label}",
                "MeanFD": np.nan,
                "NGSC_vtx": np.nan,
                "NHighResFeatures": np.nan,
                "NumFrames": np.nan,
                "ScrubbedFrames": np.nan,
                "PsiFDRegressorRow": np.nan,
                "Status": f"error: {exc}",
            })
            print(f"{row.dataset} {row.subject_id} {row.session}: failed ({exc})", flush=True)

    out = pd.DataFrame.from_records(records)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.outdir / "optional_highres_ngsc.csv", index=False)
    print(f"Wrote {args.outdir / 'optional_highres_ngsc.csv'}", flush=True)


if __name__ == "__main__":
    main()
