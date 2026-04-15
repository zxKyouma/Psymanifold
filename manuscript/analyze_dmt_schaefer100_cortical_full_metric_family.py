from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

from manuscript_paths import output_path

BASE = output_path("dmt_schaefer100_cortical")
CORE_PATH = BASE / "dmt_schaefer100_cortical_metrics" / "harmonized_results_summary.csv"
NG_PATH = BASE / "dmt_schaefer100_cortical_metrics" / "harmonized_network_to_global_distances.csv"
PAIR_PATH = BASE / "dmt_schaefer100_cortical_metrics" / "harmonized_network_procrustes_distances.csv"
EXTDIR = BASE / "dmt_schaefer100_cortical_extended_metrics"
TS_MANIFEST = BASE / "harmonized_timeseries_manifest.csv"
OUTDIR = BASE / "full_metric_family"
LEIDA_K = 6
STATE_COLS = [f"Occupancy_State{s}" for s in range(1, LEIDA_K + 1)]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bh_fdr(pvals: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(pvals, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return out
    x = arr[mask]
    order = np.argsort(x)
    ranked = x[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    out[mask] = q[inv]
    return out


def occupancy_entropy_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    return float(-(p * np.log2(p)).sum())


def weighted_state_dispersion(p: np.ndarray, centroids: np.ndarray) -> float:
    total = np.sum(p)
    if total <= 0 or not np.isfinite(total):
        return np.nan
    p = p / total
    mu = p @ centroids
    sq = ((centroids - mu) ** 2).sum(axis=1)
    return float(np.sqrt((p * sq).sum()))


def load_shared_centroids() -> np.ndarray:
    df = pd.read_csv(EXTDIR / "harmonized_shared_state_templates.csv")
    roi_cols = [c for c in df.columns if c.startswith("ROI_")]
    order = np.argsort(df["StateID"].to_numpy(dtype=int))
    return df[roi_cols].to_numpy(dtype=float)[order]


def build_attractor_metrics(dyn: pd.DataFrame) -> pd.DataFrame:
    centroids = load_shared_centroids()
    rows = []
    for _, row in dyn.iterrows():
        p = row[STATE_COLS].to_numpy(dtype=float)
        top2 = np.sort(p)[-2:]
        ent = occupancy_entropy_bits(p)
        rows.append(
            {
                **row.to_dict(),
                "DominantStateOccupancy": float(np.max(p)),
                "Top1Top2Gap": float(top2[-1] - top2[-2]),
                "OccupancyEntropy_bits": ent,
                "EffectiveNumberOfStates": float(2**ent) if np.isfinite(ent) else np.nan,
                "WeightedStateDispersion": weighted_state_dispersion(p, centroids),
            }
        )
    return pd.DataFrame(rows)


def fc_vector(timeseries_path: str) -> np.ndarray:
    ts = loadmat(timeseries_path)["time_series"]
    fc = np.corrcoef(ts, rowvar=False)
    fc = np.clip(fc, -0.999999, 0.999999)
    z = np.arctanh(fc)
    tri = np.triu_indices(z.shape[0], k=1)
    return z[tri].astype(np.float64)


def rms_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def paired_summary(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    a = np.asarray(a)[mask]
    b = np.asarray(b)[mask]
    if len(a) < 3:
        return {}
    t_stat, p_val = stats.ttest_rel(b, a)
    diff = b - a
    return {
        "N": int(len(diff)),
        "Mean_A": float(np.mean(a)),
        "Mean_B": float(np.mean(b)),
        "Delta_B_minus_A": float(np.mean(diff)),
        "TStat": float(t_stat),
        "PValue": float(p_val),
    }


def paired_subject_delta_table(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for sid, sub in df.groupby("SubjectID"):
        a = sub[sub["Condition"] == "Placebo"]
        b = sub[sub["Condition"] == "Psychedelic"]
        if len(a) != 1 or len(b) != 1:
            continue
        ra = a.iloc[0]
        rb = b.iloc[0]
        rec: dict[str, float | str] = {"SubjectID": sid}
        for metric in metric_cols:
            rec[f"Placebo_{metric}"] = float(ra.get(metric, np.nan))
            rec[f"Psychedelic_{metric}"] = float(rb.get(metric, np.nan))
            rec[f"Delta_{metric}"] = float(rb.get(metric, np.nan) - ra.get(metric, np.nan))
        rows.append(rec)
    return pd.DataFrame(rows)


def paired_metric_tests(delta_df: pd.DataFrame, metric_cols: list[str], out_name: str) -> pd.DataFrame:
    rows = []
    for metric in metric_cols:
        a = delta_df[f"Placebo_{metric}"].to_numpy(dtype=float)
        b = delta_df[f"Psychedelic_{metric}"].to_numpy(dtype=float)
        res = paired_summary(a, b)
        if not res:
            continue
        rows.append({"Metric": metric, **res})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["FDR_Q"] = bh_fdr(out["PValue"])
    out.to_csv(OUTDIR / out_name, index=False)
    return out


def main() -> None:
    ensure_dir(OUTDIR)
    core = pd.read_csv(CORE_PATH)
    ng = pd.read_csv(NG_PATH)
    pair = pd.read_csv(PAIR_PATH)
    dyn = pd.read_csv(EXTDIR / "harmonized_state_dynamics_run_metrics.csv")
    disp = pd.read_csv(EXTDIR / "harmonized_centroid_dispersion_by_run.csv")
    energy = pd.read_csv(EXTDIR / "harmonized_energy_landscape_metrics.csv")
    state_net = pd.read_csv(EXTDIR / "harmonized_state_network_correlations.csv")
    centroids = pd.read_csv(EXTDIR / "harmonized_state_centroids.csv")
    ts_manifest = pd.read_csv(TS_MANIFEST)

    join_keys = ["Dataset", "SubjectID", "Session", "Condition", "RunLabel", "FileKey"]
    dyn_core = dyn[join_keys + ["SwitchRate", "Entropy"]].copy()
    core = core.drop(columns=[c for c in ["SwitchRate", "Entropy"] if c in core.columns])
    core = core.merge(dyn_core, on=join_keys, how="left", validate="one_to_one")

    attractor = build_attractor_metrics(dyn)

    for name, df in [
        ("harmonized_results_summary.csv", core),
        ("harmonized_network_to_global_distances.csv", ng),
        ("harmonized_network_procrustes_distances.csv", pair),
        ("harmonized_state_dynamics_run_metrics.csv", dyn),
        ("harmonized_centroid_dispersion_by_run.csv", disp),
        ("harmonized_energy_landscape_metrics.csv", energy),
        ("harmonized_state_network_correlations.csv", state_net),
        ("harmonized_state_centroids.csv", centroids),
        ("harmonized_attractor_concentration_by_run.csv", attractor),
    ]:
        df.to_csv(OUTDIR / name, index=False)

    core_metrics = ["QED", "NGSC", "Q", "PC", "FC_within", "FC_between", "SwitchRate", "Entropy"]
    core_delta = paired_subject_delta_table(core, core_metrics)
    core_delta.to_csv(OUTDIR / "dmt_schaefer100_core_subject_deltas.csv", index=False)
    paired_metric_tests(core_delta, core_metrics, "dmt_schaefer100_core_paired_tests.csv")

    disp_metrics = ["RawDispersion", "PCADispersion", "PC1_VarExplained", "PC2_VarExplained", "PC3_VarExplained"]
    disp_delta = paired_subject_delta_table(disp, disp_metrics)
    disp_delta.to_csv(OUTDIR / "dmt_schaefer100_dispersion_subject_deltas.csv", index=False)
    paired_metric_tests(disp_delta, disp_metrics, "dmt_schaefer100_dispersion_paired_tests.csv")

    energy_metrics = ["EnergyMean", "EnergySD", "StationaryMassDeepWells", "StationaryMassHighPeaks", "WellSpeed", "HillSpeed", "RadiusGyration"]
    energy_delta = paired_subject_delta_table(energy, energy_metrics)
    energy_delta.to_csv(OUTDIR / "dmt_schaefer100_energy_subject_deltas.csv", index=False)
    paired_metric_tests(energy_delta, energy_metrics, "dmt_schaefer100_energy_paired_tests.csv")

    attractor_metrics = ["DominantStateOccupancy", "Top1Top2Gap", "OccupancyEntropy_bits", "EffectiveNumberOfStates", "WeightedStateDispersion"] + STATE_COLS
    attractor_delta = paired_subject_delta_table(attractor, attractor_metrics)
    attractor_delta.to_csv(OUTDIR / "dmt_schaefer100_attractor_subject_deltas.csv", index=False)
    paired_metric_tests(attractor_delta, attractor_metrics, "dmt_schaefer100_attractor_paired_tests.csv")

    default_ng = ng[ng["Network"] == "default"].copy()
    ng_delta = paired_subject_delta_table(default_ng, ["ProcrustesGlobalDist"])
    ng_delta.to_csv(OUTDIR / "dmt_schaefer100_default_network_to_global_subject_deltas.csv", index=False)
    paired_metric_tests(ng_delta, ["ProcrustesGlobalDist"], "dmt_schaefer100_default_network_to_global_paired_tests.csv")

    pair_rows = []
    pair_tests = []
    keep_pairs = sorted(pair["Pair"].dropna().astype(str).unique())[:10]
    pair_sub = pair[pair["Pair"].isin(keep_pairs)].copy()
    for pair_name, grp in pair_sub.groupby("Pair"):
        dd = paired_subject_delta_table(grp, ["ProcrustesNetDist"])
        if dd.empty:
            continue
        dd.insert(0, "Pair", pair_name)
        pair_rows.append(dd)
        test = paired_metric_tests(dd, ["ProcrustesNetDist"], "_tmp.csv")
        if not test.empty:
            test.insert(0, "Pair", pair_name)
            pair_tests.append(test)
    if pair_rows:
        pd.concat(pair_rows, ignore_index=True).to_csv(OUTDIR / "dmt_schaefer100_selected_pairwise_geometry_subject_deltas.csv", index=False)
    if pair_tests:
        out = pd.concat(pair_tests, ignore_index=True)
        out["FDR_Q"] = bh_fdr(out["PValue"])
        out.to_csv(OUTDIR / "dmt_schaefer100_selected_pairwise_geometry_paired_tests.csv", index=False)
    tmp = OUTDIR / "_tmp.csv"
    if tmp.exists():
        tmp.unlink()

    fc_cache: dict[str, np.ndarray] = {}
    wb_rows = []
    for sid, sub in ts_manifest.groupby("subject_id"):
        a = sub[sub["condition"] == "Placebo"]
        b = sub[sub["condition"] == "Psychedelic"]
        if len(a) != 1 or len(b) != 1:
            continue
        ra = a.iloc[0]
        rb = b.iloc[0]
        for row in [ra, rb]:
            ts_path = row["timeseries_path"]
            if ts_path not in fc_cache:
                fc_cache[ts_path] = fc_vector(ts_path)
        wb_rows.append(
            {
                "SubjectID": sid,
                "wholebrain_fc_change": rms_distance(fc_cache[ra["timeseries_path"]], fc_cache[rb["timeseries_path"]]),
            }
        )
    wb = pd.DataFrame(wb_rows)
    wb.to_csv(OUTDIR / "dmt_schaefer100_wholebrain_fc_change_subject.csv", index=False)
    if not wb.empty:
        vals = wb["wholebrain_fc_change"].to_numpy(dtype=float)
        pd.DataFrame(
            [
                {
                    "Metric": "wholebrain_fc_change",
                    "N": int(np.isfinite(vals).sum()),
                    "Mean": float(np.nanmean(vals)),
                    "SD": float(np.nanstd(vals, ddof=1)),
                }
            ]
        ).to_csv(OUTDIR / "dmt_schaefer100_wholebrain_fc_change_summary.csv", index=False)


if __name__ == "__main__":
    main()
