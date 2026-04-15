import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t as t_dist

from canonical_results import charm_motion_deltas, psiconnect_core_meq_join
from manuscript_paths import output_path, repo_path, require_path

OUT = output_path("geometry_behavior_focus")

NET_GLOBAL = output_path("harmonized_psychedelics_lme", "harmonized_network_to_global_distances_with_fd.csv")
PAIRWISE = output_path("harmonized_psychedelics_lme", "harmonized_network_procrustes_distances_with_fd.csv")


def fdr_bh(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return series * np.nan
    return (series - series.mean()) / sd


def fit_ols(df: pd.DataFrame, y_col: str, x_cols: list[str]):
    x = df[x_cols].astype(float).to_numpy()
    y = df[y_col].astype(float).to_numpy()
    x = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    n, p = x.shape
    dof = n - p
    if dof <= 0:
        raise ValueError("Not enough degrees of freedom")
    sigma2 = float((resid @ resid) / dof)
    xtx_inv = np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = beta / se
    pvals = 2 * t_dist.sf(np.abs(tvals), dof)
    names = ["const"] + x_cols
    return {
        "params": dict(zip(names, beta)),
        "bse": dict(zip(names, se)),
        "pvalues": dict(zip(names, pvals)),
    }


def compute_network_deltas(dataset: str, cond_a: str, cond_b: str, candidates: list[str]):
    df = pd.read_csv(NET_GLOBAL)
    df = df[df["Dataset"].eq(dataset)].copy()
    df["MeanFD_final"] = pd.to_numeric(df["MeanFD_final"], errors="coerce")
    df["ProcrustesGlobalDist"] = pd.to_numeric(df["ProcrustesGlobalDist"], errors="coerce")
    df = df[df["Network"].isin(candidates)]
    agg = (
        df.groupby(["SubjectID", "Condition", "Network"], as_index=False)
        .agg(ProcrustesGlobalDist=("ProcrustesGlobalDist", "mean"),
             MeanFD_final=("MeanFD_final", "mean"))
    )
    wide = agg.pivot(index="SubjectID", columns=["Condition", "Network"])
    wide.columns = [f"{metric}_{cond}_{net}" for metric, cond, net in wide.columns]
    wide = wide.reset_index()
    out = wide[["SubjectID"]].copy()
    for net in candidates:
        a_col = f"ProcrustesGlobalDist_{cond_a}_{net}"
        b_col = f"ProcrustesGlobalDist_{cond_b}_{net}"
        if a_col in wide.columns and b_col in wide.columns:
            out[f"Delta_NetGlobal_{net}"] = wide[b_col] - wide[a_col]
    fd_a = [c for c in wide.columns if c.startswith("MeanFD_final_") and f"_{cond_a}_" in c]
    fd_b = [c for c in wide.columns if c.startswith("MeanFD_final_") and f"_{cond_b}_" in c]
    if fd_a and fd_b:
        out["Delta_MeanFD_geom"] = wide[fd_b].mean(axis=1) - wide[fd_a].mean(axis=1)
    return out


def compute_pair_deltas(dataset: str, cond_a: str, cond_b: str, candidates: list[str]):
    df = pd.read_csv(PAIRWISE)
    df = df[df["Dataset"].eq(dataset)].copy()
    df["MeanFD_final"] = pd.to_numeric(df["MeanFD_final"], errors="coerce")
    df["ProcrustesNetDist"] = pd.to_numeric(df["ProcrustesNetDist"], errors="coerce")
    df = df[df["Pair"].isin(candidates)]
    agg = (
        df.groupby(["SubjectID", "Condition", "Pair"], as_index=False)
        .agg(ProcrustesNetDist=("ProcrustesNetDist", "mean"),
             MeanFD_final=("MeanFD_final", "mean"))
    )
    wide = agg.pivot(index="SubjectID", columns=["Condition", "Pair"])
    wide.columns = [f"{metric}_{cond}_{pair}" for metric, cond, pair in wide.columns]
    wide = wide.reset_index()
    out = wide[["SubjectID"]].copy()
    for pair in candidates:
        a_col = f"ProcrustesNetDist_{cond_a}_{pair}"
        b_col = f"ProcrustesNetDist_{cond_b}_{pair}"
        if a_col in wide.columns and b_col in wide.columns:
            out[f"Delta_Proc_{pair}"] = wide[b_col] - wide[a_col]
    return out


def run_panel(label: str, merged: pd.DataFrame, outcomes: list[str], motion_col: str):
    predictors = [c for c in merged.columns if c.startswith("Delta_NetGlobal_") or c.startswith("Delta_Proc_")]
    rows = []
    use_motion = motion_col in merged.columns and merged[motion_col].notna().any()
    for outcome in outcomes:
        for predictor in predictors:
            cols = [predictor, outcome] + ([motion_col] if use_motion else [])
            sub = merged[cols].dropna().copy()
            if len(sub) < 6:
                continue
            r, p = pearsonr(sub[predictor], sub[outcome])
            rho, ps = spearmanr(sub[predictor], sub[outcome])
            fit_cols = [predictor, motion_col] if use_motion else [predictor]
            fit = fit_ols(sub, outcome, fit_cols)
            z = sub[fit_cols + [outcome]].apply(zscore).dropna()
            beta_z = np.nan
            if len(z) >= 6:
                fit_z = fit_ols(z, outcome, fit_cols)
                beta_z = fit_z["params"].get(predictor, np.nan)
            rows.append({
                "Dataset": label,
                "Predictor": predictor,
                "Outcome": outcome,
                "N": len(sub),
                "Pearson_r": r,
                "Pearson_p": p,
                "Spearman_rho": rho,
                "Spearman_p": ps,
                "Beta": fit["params"].get(predictor, np.nan),
                "SE": fit["bse"].get(predictor, np.nan),
                "p_motion_adj": fit["pvalues"].get(predictor, np.nan),
                "Beta_z_motion_adj": beta_z,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Pearson_q_global"] = fdr_bh(out["Pearson_p"])
    out["p_motion_adj_q_global"] = fdr_bh(out["p_motion_adj"])
    out["Pearson_q_within_outcome"] = np.nan
    out["p_motion_adj_q_within_outcome"] = np.nan
    for outcome, idx in out.groupby("Outcome").groups.items():
        ix = list(idx)
        out.loc[ix, "Pearson_q_within_outcome"] = fdr_bh(out.loc[ix, "Pearson_p"])
        out.loc[ix, "p_motion_adj_q_within_outcome"] = fdr_bh(out.loc[ix, "p_motion_adj"])
    return out.sort_values(["Pearson_p", "p_motion_adj", "Outcome", "Predictor"]).reset_index(drop=True)


def load_psiconnect():
    networks = ["default", "somatomotorHand", "dorsalAttention", "cinguloopercular", "visual"]
    pairs = ["cinguloparietal-retrosplenialTemporal", "retrosplenialTemporal-salience"]
    geom = compute_network_deltas("PsiConnect_mean", "Baseline", "Psilocybin", networks)
    pair = compute_pair_deltas("PsiConnect_mean", "Baseline", "Psilocybin", pairs)
    base = psiconnect_core_meq_join()[["SubjectID", "Delta_MeanFD"]]
    meq = pd.read_csv(
        require_path("PSICONNECT_MEQ30_TSV", "PsiConnect MEQ30 questionnaire TSV"),
        sep="\t",
    ).rename(columns={"participant_id": "SubjectID"})
    asc = pd.read_csv(
        require_path("PSICONNECT_ASC11_TSV", "PsiConnect ASC11 questionnaire TSV"),
        sep="\t",
    ).rename(columns={"participant_id": "SubjectID"})
    merged = geom.merge(pair, on="SubjectID", how="outer").merge(base, on="SubjectID", how="left")
    meq_merged = merged.merge(meq[["SubjectID", "MEQ30_MEAN", "MEQ30_MYSTICAL", "MEQ30_TRANSCEND"]], on="SubjectID", how="left")
    asc_cols = [c for c in asc.columns if c.startswith("ASC11_")]
    asc_merged = merged.merge(asc[["SubjectID"] + asc_cols], on="SubjectID", how="left")
    return meq_merged, asc_merged


def load_charm():
    networks = ["frontoparietal"]
    pairs = [
        "cinguloopercular-frontoparietal",
        "frontoparietal-salience",
        "default-frontoparietal",
        "cinguloparietal-retrosplenialTemporal",
        "retrosplenialTemporal-salience",
    ]
    geom = compute_network_deltas("CHARM_Psilocybin", "Placebo", "Psychedelic", networks)
    pair = compute_pair_deltas("CHARM_Psilocybin", "Placebo", "Psychedelic", pairs)
    motion = charm_motion_deltas()

    meq = pd.read_csv(require_path("CHARM_MEQ_CSV", "CHARM questionnaire CSV"))
    mapping = pd.read_csv(repo_path("assets", "psilocybin_meq_subject_mapping.csv"))
    meq = meq.rename(columns={"Record ID:": "PS_ID", "Positive mood": "PositiveMood", "Transcendence ": "Transcendence"})
    meq = meq.merge(mapping, left_on="SubID", right_on="MEQ_ID", how="left")
    keep = ["SubjectID", "Intervention", "Mystical", "PositiveMood", "Transcendence", "Ineffability"]
    meq = meq[keep].copy()
    for col in ["Mystical", "PositiveMood", "Transcendence", "Ineffability"]:
        meq[col] = pd.to_numeric(meq[col], errors="coerce")
    meq["MEQ_mean4"] = meq[["Mystical", "PositiveMood", "Transcendence", "Ineffability"]].mean(axis=1)
    meq["Condition"] = meq["Intervention"].map({"MTP": "Placebo", "PSIL": "Psychedelic"})
    meq = meq.dropna(subset=["SubjectID", "Condition"])
    meq_wide = meq.pivot(index="SubjectID", columns="Condition")
    meq_wide.columns = [f"{a}_{b}" for a, b in meq_wide.columns]
    meq_wide = meq_wide.reset_index()
    for col in ["Mystical", "PositiveMood", "Transcendence", "Ineffability", "MEQ_mean4"]:
        meq_wide[f"Delta_{col}"] = meq_wide[f"{col}_Psychedelic"] - meq_wide[f"{col}_Placebo"]

    merged = geom.merge(pair, on="SubjectID", how="outer").merge(motion, on="SubjectID", how="left")
    merged = merged.merge(
        meq_wide[["SubjectID", "Delta_Mystical", "Delta_PositiveMood", "Delta_Transcendence", "Delta_Ineffability", "Delta_MEQ_mean4"]],
        on="SubjectID",
        how="left",
    )
    return merged


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    psi_meq, psi_asc = load_psiconnect()
    psi_meq.to_csv(OUT / "psiconnect_geometry_meq_join.csv", index=False)
    psi_asc.to_csv(OUT / "psiconnect_geometry_asc11_join.csv", index=False)
    psi_meq_res = run_panel("PsiConnect_MEQ", psi_meq, ["MEQ30_MEAN", "MEQ30_MYSTICAL", "MEQ30_TRANSCEND"], "Delta_MeanFD")
    psi_asc_outcomes = [c for c in psi_asc.columns if c.startswith("ASC11_")]
    psi_asc_res = run_panel("PsiConnect_ASC11", psi_asc, psi_asc_outcomes, "Delta_MeanFD")
    psi_meq_res.to_csv(OUT / "psiconnect_geometry_meq_results.csv", index=False)
    psi_asc_res.to_csv(OUT / "psiconnect_geometry_asc11_results.csv", index=False)

    charm = load_charm()
    charm.to_csv(OUT / "charm_geometry_meq_join.csv", index=False)
    charm_res = run_panel("CHARM_MEQ", charm, ["Delta_Mystical", "Delta_Transcendence", "Delta_MEQ_mean4"], "Delta_MeanFD")
    charm_res.to_csv(OUT / "charm_geometry_meq_results.csv", index=False)

    for name, df in [("psiconnect_geometry_meq", psi_meq_res), ("psiconnect_geometry_asc11", psi_asc_res), ("charm_geometry_meq", charm_res)]:
        if not df.empty:
            df.nsmallest(20, "Pearson_p").to_csv(OUT / f"{name}_top20.csv", index=False)


if __name__ == "__main__":
    main()
