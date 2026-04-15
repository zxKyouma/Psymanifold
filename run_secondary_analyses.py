from itertools import combinations
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("HARMONIZED_PARALLEL_EXT_OUTDIR", ROOT / "outputs" / "harmonized_psychedelics_parallel_extensions"))
OUT.mkdir(parents=True, exist_ok=True)

LME = Path(os.environ.get("HARMONIZED_LME_OUTDIR", ROOT / "outputs" / "harmonized_psychedelics_lme"))
SUMM = Path(os.environ.get("HARMONIZED_SUMMARY_OUTDIR", ROOT / "outputs" / "harmonized_psychedelics_summary"))
EXT = Path(os.environ.get("HARMONIZED_EXT_OUTDIR", ROOT / "outputs" / "harmonized_psychedelics_extended_metrics"))

CONTRASTS = {
    "PsiConnect_mean": ("Baseline", "Psilocybin"),
    "CHARM_Psilocybin": ("Placebo", "Psychedelic"),
    "LSD": ("Placebo", "Psychedelic"),
    "DMT": ("Placebo", "Psychedelic"),
}


def bh(df, p_col, out_col):
    df = df.copy()
    p = df[p_col].to_numpy(dtype=float)
    ok = np.isfinite(p)
    q = np.full(len(df), np.nan)
    if ok.any():
        pv = p[ok]
        order = np.argsort(pv)
        ranked = pv[order]
        m = len(ranked)
        adj = ranked * m / np.arange(1, m + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        adj = np.clip(adj, 0, 1)
        qvals = np.empty_like(adj)
        qvals[order] = adj
        q[ok] = qvals
    df[out_col] = q
    return df


def ols_fit(X, y):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sse = float(np.dot(resid, resid))
    rank = np.linalg.matrix_rank(X)
    n = len(y)
    df_resid = n - rank
    return beta, sse, rank, df_resid


def paired_t(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    a = np.asarray(a)[mask]
    b = np.asarray(b)[mask]
    if len(a) < 3:
        return len(a), np.nan, np.nan, np.nan
    t, p = stats.ttest_rel(a, b, nan_policy="omit")
    return len(a), float(np.nanmean(b) - np.nanmean(a)), float(t), float(p)


def build_subject_deltas():
    metrics = pd.read_csv(LME / "harmonized_results_summary_with_fd.csv")
    dyn = pd.read_csv(LME / "harmonized_state_dynamics_run_metrics_with_fd.csv")
    disp = pd.read_csv(SUMM / "harmonized_centroid_dispersion_subject_condition_means.csv")
    ng = pd.read_csv(LME / "harmonized_network_to_global_distances_with_fd.csv")
    pairs = pd.read_csv(LME / "harmonized_network_procrustes_distances_with_fd.csv")

    ng = ng[ng["Network"] == "default"].copy()
    ng = ng.groupby(["Dataset", "SubjectID", "Condition"], as_index=False)["ProcrustesGlobalDist"].mean()
    ng = ng.rename(columns={"ProcrustesGlobalDist": "Default_GlobalDist"})

    pairs = pairs[pairs["Pair"] == "default-somatomotorHand"].copy()
    pairs = pairs.groupby(["Dataset", "SubjectID", "Condition"], as_index=False)["ProcrustesNetDist"].mean()
    pairs = pairs.rename(columns={"ProcrustesNetDist": "DMN_SM_Procrustes"})

    keep = ["Dataset", "SubjectID", "Condition", "QED", "NGSC", "FC_within", "FC_between"]
    base = metrics[keep].copy()
    dyn_keep = ["Dataset", "SubjectID", "Condition", "SwitchRate", "Entropy"]
    dyn_means = dyn[dyn_keep].groupby(["Dataset", "SubjectID", "Condition"], as_index=False).mean(numeric_only=True)
    disp_keep = disp[["Dataset", "SubjectID", "Condition", "RawDispersion"]].copy()

    merged = (
        base.merge(dyn_means, on=["Dataset", "SubjectID", "Condition"], how="left")
            .merge(disp_keep, on=["Dataset", "SubjectID", "Condition"], how="left")
            .merge(ng, on=["Dataset", "SubjectID", "Condition"], how="left")
            .merge(pairs, on=["Dataset", "SubjectID", "Condition"], how="left")
    )
    return merged


def run_cross_metric():
    merged = build_subject_deltas()
    delta_rows = []
    corr_rows = []
    metric_cols = [
        "QED",
        "NGSC",
        "FC_within",
        "FC_between",
        "SwitchRate",
        "Entropy",
        "RawDispersion",
        "Default_GlobalDist",
        "DMN_SM_Procrustes",
    ]
    for dataset, (cond_a, cond_b) in CONTRASTS.items():
        sub = merged[(merged["Dataset"] == dataset) & (merged["Condition"].isin([cond_a, cond_b]))].copy()
        wide = sub.pivot_table(index="SubjectID", columns="Condition", values=metric_cols, aggfunc="mean")
        out = pd.DataFrame(index=wide.index)
        for metric in metric_cols:
            if (metric, cond_a) in wide.columns and (metric, cond_b) in wide.columns:
                out[metric] = wide[(metric, cond_b)] - wide[(metric, cond_a)]
        out = out.reset_index()
        out["Dataset"] = dataset
        out["ConditionA"] = cond_a
        out["ConditionB"] = cond_b
        delta_rows.append(out)

        usable = [m for m in metric_cols if m in out.columns]
        for m1, m2 in combinations(usable, 2):
            pair = out[[m1, m2]].dropna()
            if len(pair) < 5:
                continue
            r, p = stats.pearsonr(pair[m1], pair[m2])
            corr_rows.append({
                "Dataset": dataset,
                "ConditionA": cond_a,
                "ConditionB": cond_b,
                "MetricA": m1,
                "MetricB": m2,
                "N": len(pair),
                "PearsonR": float(r),
                "PValue": float(p),
            })

    delta_df = pd.concat(delta_rows, ignore_index=True)
    corr_df = pd.DataFrame(corr_rows)
    if not corr_df.empty:
        corr_df = corr_df.groupby("Dataset", group_keys=False).apply(lambda d: bh(d, "PValue", "FDR_Q")).reset_index(drop=True)
    delta_df.to_csv(OUT / "harmonized_cross_metric_subject_deltas.csv", index=False)
    corr_df.to_csv(OUT / "harmonized_cross_metric_coupling_stats.csv", index=False)


def run_peripherality():
    pairs = pd.read_csv(LME / "harmonized_network_procrustes_distances_with_fd.csv")
    rows = []
    for (dataset, subj, cond, run, key), grp in pairs.groupby(["Dataset", "SubjectID", "Condition", "RunLabel", "FileKey"]):
        networks = sorted(set(grp["Pair"].str.split("-").explode().dropna()))
        for network in networks:
            mask = grp["Pair"].str.startswith(network + "-") | grp["Pair"].str.endswith("-" + network)
            vals = grp.loc[mask, "ProcrustesNetDist"].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                "Dataset": dataset,
                "SubjectID": subj,
                "Condition": cond,
                "RunLabel": run,
                "FileKey": key,
                "Network": network,
                "Peripherality": vals.mean(),
            })
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "harmonized_network_peripherality_by_run.csv", index=False)

    stats_rows = []
    for dataset, (cond_a, cond_b) in CONTRASTS.items():
        sub = per[(per["Dataset"] == dataset) & (per["Condition"].isin([cond_a, cond_b]))].copy()
        means = sub.groupby(["SubjectID", "Condition", "Network"], as_index=False)["Peripherality"].mean()
        for network, grp in means.groupby("Network"):
            wide = grp.pivot_table(index="SubjectID", columns="Condition", values="Peripherality", aggfunc="mean")
            if cond_a not in wide.columns or cond_b not in wide.columns:
                continue
            n, delta, t, p = paired_t(wide[cond_a], wide[cond_b])
            stats_rows.append({
                "Dataset": dataset,
                "ConditionA": cond_a,
                "ConditionB": cond_b,
                "Network": network,
                "NPaired": n,
                "MeanA": float(np.nanmean(wide[cond_a])),
                "MeanB": float(np.nanmean(wide[cond_b])),
                "Delta_B_minus_A": delta,
                "TStat": t,
                "PValue": p,
            })
    out = pd.DataFrame(stats_rows)
    if not out.empty:
        out = out.groupby("Dataset", group_keys=False).apply(lambda d: bh(d, "PValue", "FDR_Q")).reset_index(drop=True)
    out.to_csv(OUT / "harmonized_network_peripherality_paired_stats.csv", index=False)


def build_state_design(sub, include_threeway):
    cols = []

    subject = pd.get_dummies(sub["SubjectID"], prefix="S", drop_first=True, dtype=float)
    cond = (sub["Condition"] == sub["Condition"].cat.categories[-1]).astype(float).rename("ConditionB").to_frame()
    state = pd.get_dummies(sub["StateID"].astype(str), prefix="State", drop_first=True, dtype=float)
    network = pd.get_dummies(sub["Network"].astype(str), prefix="Net", drop_first=True, dtype=float)

    cols.extend([subject, cond, state, network])

    if not state.empty:
        cols.append(state.mul(cond.iloc[:, 0], axis=0).add_prefix("CondX"))
    if not network.empty:
        cols.append(network.mul(cond.iloc[:, 0], axis=0).add_prefix("CondX"))
    if not state.empty and not network.empty:
        inter_sn = pd.DataFrame(index=sub.index)
        for s in state.columns:
            for n in network.columns:
                inter_sn[f"{s}__{n}"] = state[s] * network[n]
        cols.append(inter_sn)
        if include_threeway:
            inter_csn = pd.DataFrame(index=sub.index)
            for s in state.columns:
                for n in network.columns:
                    inter_csn[f"Cond__{s}__{n}"] = cond.iloc[:, 0] * state[s] * network[n]
            cols.append(inter_csn)

    X = pd.concat(cols, axis=1)
    if "MeanFD" in sub.columns and sub["MeanFD"].notna().any():
        X["MeanFDc"] = sub["MeanFD"] - sub["MeanFD"].mean()
    X.insert(0, "Intercept", 1.0)
    return X


def run_state_structure():
    raise RuntimeError(
        "State-specific omnibus testing is handled in fit_extension_mixed_effects.m. "
        "This older OLS helper is kept only for reference and is not part of the public inferential path."
    )


def main():
    run_cross_metric()
    run_peripherality()
    print(f"Wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
