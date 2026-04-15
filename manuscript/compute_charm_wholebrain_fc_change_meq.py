import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import linregress, pearsonr, ttest_rel

from manuscript_paths import configured_path, output_path, require_path

MEQ_CSV = require_path("CHARM_MEQ_CSV", "CHARM questionnaire CSV")
OUTDIR = output_path("charm_fc_change_meq")


SUB_TO_PS = {
    "sub-1": "PS03",
    "sub-2": "PS02",
    "sub-3": "PS16",
    "sub-4": "PS18",
    "sub-5": "PS19",
    "sub-6": "PS21",
    "sub-7": "PS24",
}

MEQ_TO_PS = {
    "P1": "PS03",
    "P2": "PS02",
    "P3": "PS16",
    "P4": "PS18",
    "P5": "PS19",
    "P6": "PS21",
    "P7": "PS24",
}


def load_charm_manifest() -> pd.DataFrame:
    csv_path = configured_path(
        "HARMONIZED_TS_MANIFEST",
        default_rel="outputs/harmonized_psychedelics/harmonized_timeseries_manifest.csv",
    )
    if csv_path is None or not csv_path.exists():
        csv_path = configured_path(
            "HARMONIZED_TS_MANIFEST",
            default_rel="outputs/harmonized_psychedelics/harmonized_timeseries_manifest.tsv",
        )
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(
            "Missing harmonized timeseries manifest. Run the public core pipeline first or set HARMONIZED_TS_MANIFEST."
        )
    df = pd.read_csv(csv_path)
    df = df[
        (df["dataset"] == "CHARM_Psilocybin")
        & (df["condition"].isin(["Baseline", "Placebo", "Psychedelic"]))
    ].copy()
    df["PS_ID"] = df["subject_id"].map(SUB_TO_PS)
    return df


def load_meq() -> pd.DataFrame:
    meq = pd.read_csv(MEQ_CSV)
    meq.columns = [c.strip() for c in meq.columns]
    meq = meq.rename(
        columns={
            "SubID": "MEQ_ID",
            "Intervention": "Condition",
            "Positive mood": "PositiveMood",
            "Transcendence ": "Transcendence",
        }
    )
    meq["MEQ_ID"] = meq["MEQ_ID"].astype(str).str.strip().str.upper()
    meq["PS_ID"] = meq["MEQ_ID"].map(MEQ_TO_PS)
    meq["Condition"] = meq["Condition"].astype(str)
    meq.loc[meq["Condition"].eq("MTP"), "Condition"] = "Placebo"
    meq.loc[meq["Condition"].str.contains("PSIL", na=False), "Condition"] = "Psychedelic"
    meq = meq[meq["PS_ID"].notna()].copy()

    for col in ["Mystical", "PositiveMood", "Transcendence", "Ineffability"]:
        meq[col] = pd.to_numeric(meq[col], errors="coerce")

    meq["MEQ_mean4"] = meq[["Mystical", "PositiveMood", "Transcendence", "Ineffability"]].mean(axis=1)

    # If PSIL2 rows exist, keep the strongest psychedelic row for each subject/condition.
    meq = meq.sort_values(["PS_ID", "Condition", "Mystical"], ascending=[True, True, False])
    meq = meq.drop_duplicates(["PS_ID", "Condition"])
    return meq


def fisher_z_fc_vector(timeseries_path: str) -> np.ndarray:
    ts = loadmat(timeseries_path)["time_series"]
    fc = np.corrcoef(ts, rowvar=False)
    fc = np.clip(fc, -0.999999, 0.999999)
    z = np.arctanh(fc)
    tri = np.triu_indices(z.shape[0], k=1)
    return z[tri].astype(np.float64)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    charm = load_charm_manifest()
    meq = load_meq()

    fc_cache: dict[tuple[str, str, str], np.ndarray] = {}
    for row in charm.itertuples(index=False):
        fc_cache[(row.subject_id, row.session, row.condition)] = fisher_z_fc_vector(row.timeseries_path)

    rows: list[dict[str, object]] = []
    acute = charm[charm["condition"].isin(["Placebo", "Psychedelic"])].copy()
    for row in acute.itertuples(index=False):
        acute_vec = fc_cache[(row.subject_id, row.session, row.condition)]
        baseline_rows = charm[
            (charm["subject_id"] == row.subject_id) & (charm["condition"] == "Baseline")
        ]
        dists = []
        for brow in baseline_rows.itertuples(index=False):
            bvec = fc_cache[(brow.subject_id, brow.session, brow.condition)]
            dists.append(np.sqrt(np.mean((acute_vec - bvec) ** 2)))

        rows.append(
            {
                "subject_id": row.subject_id,
                "PS_ID": row.PS_ID,
                "acute_session": row.session,
                "Condition": row.condition,
                "wholebrain_fc_change": float(np.mean(dists)),
                "n_baseline_refs": len(dists),
            }
        )

    acute_df = pd.DataFrame(rows).merge(
        meq[
            [
                "PS_ID",
                "Condition",
                "Mystical",
                "PositiveMood",
                "Transcendence",
                "Ineffability",
                "MEQ_mean4",
            ]
        ],
        on=["PS_ID", "Condition"],
        how="left",
    )
    acute_df.to_csv(OUTDIR / "acute_wholebrain_fc_change_meq.csv", index=False)

    paired = (
        acute_df.pivot_table(
            index="subject_id",
            columns="Condition",
            values="wholebrain_fc_change",
            aggfunc="mean",
        )
        .dropna()
        .reset_index()
    )
    if {"Placebo", "Psychedelic"}.issubset(paired.columns):
        placebo = paired["Placebo"].to_numpy(dtype=float)
        psychedelic = paired["Psychedelic"].to_numpy(dtype=float)
        t_stat, p_val = ttest_rel(placebo, psychedelic)
        pd.DataFrame(
            [
                {
                    "N": int(len(paired)),
                    "Placebo_mean": float(np.mean(placebo)),
                    "Psychedelic_mean": float(np.mean(psychedelic)),
                    "Delta_mean": float(np.mean(psychedelic - placebo)),
                    "paired_t": float(t_stat),
                    "paired_p": float(p_val),
                }
            ]
        ).to_csv(OUTDIR / "acute_wholebrain_fc_change_paired_test.csv", index=False)

    stats_rows = []
    for group_name, group_df in [("AllAcute", acute_df)] + list(acute_df.groupby("Condition")):
        for outcome in ["MEQ_mean4", "Mystical", "PositiveMood", "Transcendence", "Ineffability"]:
            mask = group_df["wholebrain_fc_change"].notna() & group_df[outcome].notna()
            x = group_df.loc[mask, "wholebrain_fc_change"].to_numpy()
            y = group_df.loc[mask, outcome].to_numpy()
            if len(x) < 3:
                continue
            r, p = pearsonr(x, y)
            fit = linregress(x, y)
            stats_rows.append(
                {
                    "Group": group_name,
                    "Outcome": outcome,
                    "N": len(x),
                    "Pearson_r": r,
                    "Pearson_r2": r * r,
                    "Pearson_p": p,
                    "Slope": fit.slope,
                    "Intercept": fit.intercept,
                    "Linreg_p": fit.pvalue,
                }
            )

    pd.DataFrame(stats_rows).to_csv(OUTDIR / "wholebrain_fc_change_meq_correlations.csv", index=False)


if __name__ == "__main__":
    main()
