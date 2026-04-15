from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import ttest_rel

from manuscript_paths import configured_path, output_path, require_path

OUTDIR = output_path("wholebrain_fc_change")


def manifest_path():
    csv_path = configured_path(
        "HARMONIZED_TS_MANIFEST",
        default_rel="outputs/harmonized_psychedelics/harmonized_timeseries_manifest.csv",
    )
    if csv_path is not None and csv_path.exists():
        return csv_path
    tsv_path = configured_path(
        "HARMONIZED_TS_MANIFEST",
        default_rel="outputs/harmonized_psychedelics/harmonized_timeseries_manifest.tsv",
    )
    if tsv_path is not None and tsv_path.exists():
        return tsv_path
    raise FileNotFoundError(
        "Missing harmonized timeseries manifest. Run the public core pipeline first or set HARMONIZED_TS_MANIFEST."
    )


def fc_vector_from_ts(ts: np.ndarray) -> np.ndarray:
    fc = np.corrcoef(ts, rowvar=False)
    fc = np.clip(fc, -0.999999, 0.999999)
    z = np.arctanh(fc)
    tri = np.triu_indices(z.shape[0], k=1)
    return z[tri].astype(np.float64)


def fc_vector(timeseries_path: str) -> np.ndarray:
    ts = loadmat(timeseries_path)["time_series"]
    return fc_vector_from_ts(ts)


def rms_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a - b
    t_stat, p_val = ttest_rel(a, b)
    return {
        "N": int(len(diff)),
        "mean_a": float(np.mean(a)),
        "sd_a": float(np.std(a, ddof=1)),
        "mean_b": float(np.mean(b)),
        "sd_b": float(np.std(b, ddof=1)),
        "mean_diff_a_minus_b": float(np.mean(diff)),
        "sd_diff_a_minus_b": float(np.std(diff, ddof=1)),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohens_dz": float(np.mean(diff) / np.std(diff, ddof=1)),
    }


def psi_split_half_analysis(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for subject_id, sdf in manifest.groupby("subject_id"):
        base = sdf[sdf["condition"] == "Baseline"]
        drug = sdf[sdf["condition"] == "Psilocybin"]
        if len(base) == 0 or len(drug) == 0:
            continue

        base_row = next(base.itertuples(index=False))
        drug_row = next(drug.itertuples(index=False))
        base_ts = loadmat(base_row.timeseries_path)["time_series"]
        drug_ts = loadmat(drug_row.timeseries_path)["time_series"]

        b_mid = base_ts.shape[0] // 2
        d_mid = drug_ts.shape[0] // 2
        b1, b2 = base_ts[:b_mid, :], base_ts[b_mid:, :]
        d1, d2 = drug_ts[:d_mid, :], drug_ts[d_mid:, :]

        b1v, b2v = fc_vector_from_ts(b1), fc_vector_from_ts(b2)
        d1v, d2v = fc_vector_from_ts(d1), fc_vector_from_ts(d2)

        baseline_repeat = rms_distance(b1v, b2v)
        psychedelic_repeat = rms_distance(d1v, d2v)
        psychedelic_vs_baseline = float(
            np.mean(
                [
                    rms_distance(b1v, d1v),
                    rms_distance(b1v, d2v),
                    rms_distance(b2v, d1v),
                    rms_distance(b2v, d2v),
                ]
            )
        )

        rows.append(
            {
                "dataset": "PsiConnect_mean",
                "subject_id": subject_id,
                "baseline_repeat": baseline_repeat,
                "psychedelic_repeat": psychedelic_repeat,
                "within_mean": float(np.mean([baseline_repeat, psychedelic_repeat])),
                "psychedelic_vs_baseline": psychedelic_vs_baseline,
                "baseline_mean_fd": float(base_row.mean_fd),
                "psychedelic_mean_fd": float(drug_row.mean_fd),
            }
        )

    detail = pd.DataFrame(rows)
    tests = [
        {
            "dataset": "PsiConnect_mean",
            "comparison": "psychedelic_vs_baseline_vs_baseline_repeat",
            **paired_stats(
                detail["psychedelic_vs_baseline"].to_numpy(dtype=float),
                detail["baseline_repeat"].to_numpy(dtype=float),
            ),
        },
        {
            "dataset": "PsiConnect_mean",
            "comparison": "psychedelic_vs_baseline_vs_within_mean",
            **paired_stats(
                detail["psychedelic_vs_baseline"].to_numpy(dtype=float),
                detail["within_mean"].to_numpy(dtype=float),
            ),
        },
    ]
    return detail, pd.DataFrame(tests)


def lsd_condition_analysis(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fc_cache = {}
    for row in manifest.itertuples(index=False):
        fc_cache[(row.subject_id, row.condition, row.run_label)] = fc_vector(row.timeseries_path)

    rows = []
    for subject_id, sdf in manifest.groupby("subject_id"):
        placebo = list(sdf[sdf["condition"] == "Placebo"].itertuples(index=False))
        psychedelic = list(sdf[sdf["condition"] == "Psychedelic"].itertuples(index=False))
        if len(placebo) < 2 or len(psychedelic) < 2:
            continue

        placebo_repeat = []
        psychedelic_repeat = []
        cross = []

        for a, b in combinations(placebo, 2):
            placebo_repeat.append(
                rms_distance(
                    fc_cache[(subject_id, a.condition, a.run_label)],
                    fc_cache[(subject_id, b.condition, b.run_label)],
                )
            )
        for a, b in combinations(psychedelic, 2):
            psychedelic_repeat.append(
                rms_distance(
                    fc_cache[(subject_id, a.condition, a.run_label)],
                    fc_cache[(subject_id, b.condition, b.run_label)],
                )
            )
        for a, b in product(placebo, psychedelic):
            cross.append(
                rms_distance(
                    fc_cache[(subject_id, a.condition, a.run_label)],
                    fc_cache[(subject_id, b.condition, b.run_label)],
                )
            )

        rows.append(
            {
                "dataset": "LSD",
                "subject_id": subject_id,
                "placebo_repeat": float(np.mean(placebo_repeat)),
                "psychedelic_repeat": float(np.mean(psychedelic_repeat)),
                "within_mean": float(np.mean(placebo_repeat + psychedelic_repeat)),
                "psychedelic_vs_placebo": float(np.mean(cross)),
            }
        )

    detail = pd.DataFrame(rows)
    tests = [
        {
            "dataset": "LSD",
            "comparison": "psychedelic_vs_placebo_vs_placebo_repeat",
            **paired_stats(
                detail["psychedelic_vs_placebo"].to_numpy(dtype=float),
                detail["placebo_repeat"].to_numpy(dtype=float),
            ),
        },
        {
            "dataset": "LSD",
            "comparison": "psychedelic_vs_placebo_vs_within_mean",
            **paired_stats(
                detail["psychedelic_vs_placebo"].to_numpy(dtype=float),
                detail["within_mean"].to_numpy(dtype=float),
            ),
        },
    ]
    return detail, pd.DataFrame(tests)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path())

    psi = manifest[manifest["dataset"] == "PsiConnect_mean"].copy()
    psi_detail, psi_tests = psi_split_half_analysis(psi)
    psi_detail.to_csv(OUTDIR / "psiconnect_splithalf_wholebrain_fc_change_detail.csv", index=False)
    psi_tests.to_csv(OUTDIR / "psiconnect_splithalf_wholebrain_fc_change_tests.csv", index=False)
    try:
        meq = pd.read_csv(
            require_path("PSICONNECT_MEQ30_TSV", "PsiConnect MEQ30 questionnaire TSV"),
            sep="\t",
        ).rename(columns={"participant_id": "subject_id"})
        join = psi_detail.merge(
            meq[
                [
                    "subject_id",
                    "MEQ30_MEAN",
                    "MEQ30_MYSTICAL",
                    "MEQ30_POSITIVE",
                    "MEQ30_TRANSCEND",
                    "MEQ30_INEFFABILITY",
                ]
            ],
            on="subject_id",
            how="left",
        )
        join["wholebrain_fc_change"] = join["psychedelic_vs_baseline"]
        join["delta_mean_fd"] = join["psychedelic_mean_fd"] - join["baseline_mean_fd"]
        join.to_csv(OUTDIR / "psiconnect_wholebrain_fc_change_meq_join.csv", index=False)
    except FileNotFoundError:
        pass

    lsd = manifest[manifest["dataset"] == "LSD"].copy()
    lsd_detail, lsd_tests = lsd_condition_analysis(lsd)
    lsd_detail.to_csv(OUTDIR / "lsd_condition_wholebrain_fc_change_detail.csv", index=False)
    lsd_tests.to_csv(OUTDIR / "lsd_condition_wholebrain_fc_change_tests.csv", index=False)


if __name__ == "__main__":
    main()
