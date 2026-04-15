#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_config import DEFAULT_MANIFEST_DIR


REQUIRED_COLUMNS = [
    "dataset",
    "subject_id",
    "session",
    "condition",
    "run_label",
    "input_type",
    "input_path",
    "confounds_path",
    "anatomy_path",
    "existing_timeseries_path",
    "mean_fd",
    "notes",
]

DATASET_RENAME_MAP = {
    "PsiConnect_mean": "PsiConnect",
    "CHARM_Psilocybin": "Psilocybin",
    "Siegel_et_al_psilocybin": "Psilocybin",
    "DMT": "DMT",
}


def add_dmt_rows(dmt_root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(dmt_root.glob("LongS*.mat")):
        stem = p.stem
        if stem.endswith("DMT"):
            condition = "Psychedelic"
        elif stem.endswith("PCB"):
            condition = "Placebo"
        else:
            continue
        rows.append(
            {
                "dataset": "DMT",
                "subject_id": stem[:-3],
                "session": "ses-01",
                "condition": condition,
                "run_label": "task-rest",
                "input_type": "precomputed_roi_timeseries_mat",
                "input_path": str(p),
                "confounds_path": "",
                "anatomy_path": "",
                "existing_timeseries_path": "",
                "mean_fd": np.nan,
                "notes": "DMT AAL-112 time series from MATLAB .mat; reduced pathway without network-mapping analyses",
            }
        )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def add_lsd_rows(lsd_root: Path) -> pd.DataFrame:
    rows = []
    pattern = "sub-*/ses-*/func/*run-0[13]_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    for p in sorted(lsd_root.glob(pattern)):
        rel = p.relative_to(lsd_root)
        subj, ses, _, fname = rel.parts
        run = "run-01" if "run-01" in fname else "run-03"
        conf = p.with_name(
            fname.replace(
                "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                "_desc-confounds_timeseries.tsv",
            )
        )
        mean_fd = np.nan
        if conf.exists():
            try:
                cdf = pd.read_csv(conf, sep="\t", usecols=["framewise_displacement"])
                s = pd.to_numeric(cdf["framewise_displacement"], errors="coerce").dropna()
                mean_fd = float(s.mean()) if len(s) else np.nan
            except Exception:
                pass
        rows.append(
            {
                "dataset": "LSD",
                "subject_id": subj,
                "session": ses,
                "condition": "Psychedelic" if ses == "ses-LSD" else "Placebo",
                "run_label": run,
                "input_type": "fmriprep_desc_preproc_bold",
                "input_path": str(p),
                "confounds_path": str(conf) if conf.exists() else "",
                "anatomy_path": "",
                "existing_timeseries_path": "",
                "mean_fd": mean_fd,
                "notes": "LSD fMRIPrep derivative; run-01/run-03 retained for comparability",
            }
        )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a harmonized psychedelics manifest.")
    parser.add_argument("--base-manifest", type=Path, required=True, help="CSV manifest containing PsiConnect/CHARM rows.")
    parser.add_argument("--lsd-root", type=Path, default=None, help="Optional LSD derivative root to append.")
    parser.add_argument("--dmt-root", type=Path, default=None, help="Optional DMT .mat root to append.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_MANIFEST_DIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.base_manifest).fillna("")
    missing = [c for c in REQUIRED_COLUMNS if c not in base.columns]
    if missing:
        raise ValueError(f"Base manifest missing columns: {missing}")

    all_df = base[REQUIRED_COLUMNS].copy()
    all_df["dataset"] = all_df["dataset"].replace(DATASET_RENAME_MAP)
    if args.lsd_root:
        lsd = add_lsd_rows(args.lsd_root)
        all_df = pd.concat([all_df, lsd], ignore_index=True)
    if args.dmt_root:
        dmt = add_dmt_rows(args.dmt_root)
        dmt["dataset"] = dmt["dataset"].replace(DATASET_RENAME_MAP)
        all_df = pd.concat([all_df, dmt], ignore_index=True)

    out = args.outdir / "harmonized_manifest.csv"
    all_df.to_csv(out, index=False)
    summary = (
        all_df.groupby("dataset")
        .agg(
            n_rows=("dataset", "size"),
            n_subjects=("subject_id", "nunique"),
            n_conditions=("condition", "nunique"),
            with_input=("input_path", lambda s: int(sum(bool(x) for x in s))),
            with_confounds=("confounds_path", lambda s: int(sum(bool(x) for x in s))),
            with_existing_ts=("existing_timeseries_path", lambda s: int(sum(bool(x) for x in s))),
        )
        .reset_index()
    )
    summary.to_csv(args.outdir / "harmonized_manifest_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
