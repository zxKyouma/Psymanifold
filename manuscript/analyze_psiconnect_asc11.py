import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t as t_dist

from canonical_results import compute_fc_global_for_psiconnect, psiconnect_core_meq_join
from manuscript_paths import output_path, require_path


ASC_PATH = require_path("PSICONNECT_ASC11_TSV", "PsiConnect ASC11 questionnaire TSV")
OUT = output_path("asc11_analysis")


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
        raise ValueError("Not enough degrees of freedom for OLS")
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


def load_asc():
    asc = pd.read_csv(ASC_PATH, sep="\t")
    asc = asc.rename(columns={"participant_id": "SubjectID"})
    outcome_cols = [c for c in asc.columns if c.startswith("ASC11_")]
    keep = ["SubjectID"] + outcome_cols
    return asc[keep].copy(), outcome_cols


def merge_asc(df: pd.DataFrame, asc: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "SubjectID" not in out.columns:
        if "subject_id" in out.columns:
            out = out.rename(columns={"subject_id": "SubjectID"})
        elif "participant_id" in out.columns:
            out = out.rename(columns={"participant_id": "SubjectID"})
        else:
            raise KeyError("No compatible subject id column found")
    return out.merge(asc, on="SubjectID", how="left")


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return series * np.nan
    return (series - series.mean()) / sd


def build_candidate_panel() -> pd.DataFrame:
    base = pd.read_csv(output_path("missing_meq_panels", "psiconnect_sharedstate_meq_join.csv"))
    attr = pd.read_csv(output_path("attractor_concentration", "acute_delta_attractor_concentration.csv"))
    attr = attr[attr["Dataset"].eq("PsiConnect")].copy()
    attr = attr[
        [
            "SubjectID",
            "Delta_DominantStateOccupancy",
            "Delta_OccupancyEntropy_bits",
            "Delta_WeightedStateDispersion",
            "DeltaQED",
            "DeltaSwitchRate",
            "DeltaEntropy",
        ]
    ]

    net = pd.read_csv(output_path("harmonized_psychedelics_lme", "harmonized_network_to_global_distances_with_fd.csv"))
    net = net[net["Dataset"].eq("PsiConnect_mean") & net["Network"].isin(["somatomotorHand", "dorsalAttention", "cinguloopercular", "default", "visual"])].copy()
    net = net.groupby(["SubjectID", "Condition", "Network"], as_index=False)["ProcrustesGlobalDist"].mean()
    net_wide = net.pivot(index="SubjectID", columns=["Condition", "Network"], values="ProcrustesGlobalDist").reset_index()
    net_wide.columns = ["SubjectID" if c == ("", "") else f"{c[0]}_{c[1]}" for c in net_wide.columns]
    geom = net_wide[["SubjectID"]].copy()
    for net_name in ["somatomotorHand", "dorsalAttention", "cinguloopercular", "default", "visual"]:
        geom[f"Delta_NetGlobal_{net_name}"] = net_wide[f"Psilocybin_{net_name}"] - net_wide[f"Baseline_{net_name}"]

    pair = pd.read_csv(output_path("harmonized_psychedelics_lme", "harmonized_network_procrustes_distances_with_fd.csv"))
    pair = pair[pair["Dataset"].eq("PsiConnect_mean") & pair["Pair"].isin([
        "cinguloparietal-retrosplenialTemporal",
        "retrosplenialTemporal-salience",
        "auditory-somatomotorMouth",
        "default-salience",
    ])].copy()
    pair = pair.groupby(["SubjectID", "Condition", "Pair"], as_index=False)["ProcrustesNetDist"].mean()
    pair_wide = pair.pivot(index="SubjectID", columns=["Condition", "Pair"], values="ProcrustesNetDist").reset_index()
    pair_wide.columns = ["SubjectID" if c == ("", "") else f"{c[0]}_{c[1]}" for c in pair_wide.columns]
    pair_out = pair_wide[["SubjectID"]].copy()
    for pair_name in [
        "cinguloparietal-retrosplenialTemporal",
        "retrosplenialTemporal-salience",
        "auditory-somatomotorMouth",
        "default-salience",
    ]:
        pair_out[f"Delta_Proc_{pair_name}"] = pair_wide[f"Psilocybin_{pair_name}"] - pair_wide[f"Baseline_{pair_name}"]

    energy = pd.read_csv(output_path("harmonized_psychedelics_summary", "harmonized_energy_subject_condition_means.csv"))
    energy = energy[energy["Dataset"].eq("PsiConnect_mean")].copy()
    energy = energy.groupby(["SubjectID", "Condition"], as_index=False)[["EnergySD", "WellSpeed", "HillSpeed", "RadiusGyration"]].mean()
    ewide = energy.pivot(index="SubjectID", columns="Condition").reset_index()
    ewide.columns = ["SubjectID" if c == ("", "") else f"{c[0]}_{c[1]}" for c in ewide.columns]
    eout = ewide[["SubjectID"]].copy()
    for metric in ["EnergySD", "WellSpeed", "HillSpeed", "RadiusGyration"]:
        eout[f"Delta_{metric}"] = ewide[f"{metric}_Psilocybin"] - ewide[f"{metric}_Baseline"]

    merged = base.merge(attr, on="SubjectID", how="left").merge(geom, on="SubjectID", how="left").merge(pair_out, on="SubjectID", how="left").merge(eout, on="SubjectID", how="left")
    sub = merged[["Delta_DominantStateOccupancy", "Delta_OccupancyEntropy_bits", "Delta_WeightedStateDispersion"]].copy()
    sub["Delta_OccupancyEntropy_bits"] = -sub["Delta_OccupancyEntropy_bits"]
    sub["Delta_WeightedStateDispersion"] = -sub["Delta_WeightedStateDispersion"]
    merged["AttractorComposite"] = sub.apply(zscore).mean(axis=1)
    return merged


def analyze_panel(df: pd.DataFrame, predictors: list[str], outcomes: list[str], motion_col: str, panel: str):
    rows = []
    for predictor in predictors:
        if predictor not in df.columns:
            continue
        for outcome in outcomes:
            if outcome not in df.columns:
                continue
            cols = [predictor, outcome]
            if motion_col in df.columns:
                cols.append(motion_col)
            sub = df[cols].dropna().copy()
            if len(sub) < 6:
                continue

            r, p = pearsonr(sub[predictor], sub[outcome])
            rho, p_s = spearmanr(sub[predictor], sub[outcome])
            row = {
                "Panel": panel,
                "Predictor": predictor,
                "Outcome": outcome,
                "N": len(sub),
                "Pearson_r": r,
                "Pearson_p": p,
                "Spearman_rho": rho,
                "Spearman_p": p_s,
            }

            if motion_col in sub.columns:
                fit = fit_ols(sub, outcome, [predictor, motion_col])
                z = sub[[predictor, motion_col, outcome]].apply(zscore).dropna()
                row.update(
                    {
                        "Beta": fit["params"].get(predictor, np.nan),
                        "SE": fit["bse"].get(predictor, np.nan),
                        "p_motion_adj": fit["pvalues"].get(predictor, np.nan),
                        "Beta_motion": fit["params"].get(motion_col, np.nan),
                        "p_motion": fit["pvalues"].get(motion_col, np.nan),
                    }
                )
                if len(z) >= 6:
                    fit_z = fit_ols(z, outcome, [predictor, motion_col])
                    row["Beta_z_motion_adj"] = fit_z["params"].get(predictor, np.nan)
                else:
                    row["Beta_z_motion_adj"] = np.nan

            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["Pearson_q"] = fdr_bh(out["Pearson_p"])
        if "p_motion_adj" in out.columns:
            mask = out["p_motion_adj"].notna()
            out.loc[mask, "p_motion_adj_q"] = fdr_bh(out.loc[mask, "p_motion_adj"])
        out = out.sort_values(["Pearson_p", "p_motion_adj", "Outcome", "Predictor"]).reset_index(drop=True)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    asc, outcomes = load_asc()
    asc.to_csv(OUT / "psiconnect_asc11_source.csv", index=False)

    panels = [
        {
            "name": "core_metrics",
            "df": psiconnect_core_meq_join(),
            "predictors": ["Delta_QED", "Delta_NGSC", "Delta_FC_within", "Delta_NGSC_vtx", "Delta_DMN_SM"],
            "motion": "Delta_MeanFD",
        },
        {
            "name": "fc_global",
            "df": compute_fc_global_for_psiconnect(),
            "predictors": ["Delta_FC_global"],
            "motion": "Delta_MeanFD",
        },
        {
            "name": "wholebrain_fc_change",
            "path": output_path("wholebrain_fc_change", "psiconnect_wholebrain_fc_change_meq_join.csv"),
            "predictors": ["wholebrain_fc_change"],
            "motion": "delta_mean_fd",
        },
        {
            "name": "sharedstate_metrics",
            "path": output_path("missing_meq_panels", "psiconnect_sharedstate_meq_join.csv"),
            "predictors": [
                "Delta_State1_Occupancy",
                "Delta_State2_Occupancy",
                "Delta_State3_Occupancy",
                "Delta_State4_Occupancy",
                "Delta_State5_Occupancy",
                "Delta_State6_Occupancy",
                "Delta_SwitchRate_group",
                "DeltaDispersion",
            ],
            "motion": "Delta_MeanFD",
        },
        {
            "name": "candidate_panel",
            "df": build_candidate_panel(),
            "predictors": [
                "Delta_NetGlobal_somatomotorHand",
                "Delta_NetGlobal_dorsalAttention",
                "Delta_NetGlobal_cinguloopercular",
                "Delta_NetGlobal_default",
                "Delta_NetGlobal_visual",
                "Delta_Proc_cinguloparietal-retrosplenialTemporal",
                "Delta_Proc_retrosplenialTemporal-salience",
                "Delta_Proc_auditory-somatomotorMouth",
                "Delta_Proc_default-salience",
                "AttractorComposite",
                "Delta_DominantStateOccupancy",
                "Delta_OccupancyEntropy_bits",
                "Delta_WeightedStateDispersion",
                "DeltaQED",
                "DeltaSwitchRate",
                "DeltaEntropy",
                "Delta_EnergySD",
                "Delta_WellSpeed",
                "Delta_HillSpeed",
                "Delta_RadiusGyration",
            ],
            "motion": "Delta_MeanFD",
        },
    ]

    results = []
    for panel in panels:
        joined = merge_asc(panel["df"].copy(), asc)
        joined.to_csv(OUT / f"{panel['name']}_asc11_join.csv", index=False)
        res = analyze_panel(joined, panel["predictors"], outcomes, panel["motion"], panel["name"])
        res.to_csv(OUT / f"{panel['name']}_asc11_results.csv", index=False)
        if not res.empty:
            top = res.nsmallest(15, "Pearson_p").copy()
            top["Panel"] = panel["name"]
            top.to_csv(OUT / f"{panel['name']}_asc11_top15.csv", index=False)
            results.append(res)

    if results:
        all_res = pd.concat(results, ignore_index=True)
        all_res["Global_Pearson_q"] = fdr_bh(all_res["Pearson_p"])
        mask = all_res["p_motion_adj"].notna()
        all_res.loc[mask, "Global_p_motion_adj_q"] = fdr_bh(all_res.loc[mask, "p_motion_adj"])
        all_res = all_res.sort_values(["Pearson_p", "p_motion_adj", "Panel", "Outcome", "Predictor"]).reset_index(drop=True)
        all_res.to_csv(OUT / "psiconnect_asc11_all_results.csv", index=False)
        all_res.nsmallest(30, "Pearson_p").to_csv(OUT / "psiconnect_asc11_top30_overall.csv", index=False)


if __name__ == "__main__":
    main()
