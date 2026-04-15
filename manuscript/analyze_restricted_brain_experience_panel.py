import numpy as np
import pandas as pd
from scipy import stats

from manuscript_paths import output_path

OUTDIR = output_path("restricted_brain_experience_panel")
OUTDIR.mkdir(parents=True, exist_ok=True)


def fdr_bh(pvals):
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


def raw_and_adjusted(df, predictors, outcomes, motion_col, panel_label):
    rows = []
    for predictor in predictors:
        for outcome in outcomes:
            cols = [predictor, outcome]
            if motion_col in df.columns:
                cols.append(motion_col)
            sub = df[cols].dropna().copy()
            if len(sub) < 4:
                continue

            x = sub[predictor].to_numpy(dtype=float)
            y = sub[outcome].to_numpy(dtype=float)
            r, p = stats.pearsonr(x, y)
            rho, p_s = stats.spearmanr(x, y)

            row = {
                "Panel": panel_label,
                "Predictor": predictor,
                "Outcome": outcome,
                "N": len(sub),
                "Pearson_r": float(r),
                "Pearson_p": float(p),
                "Spearman_rho": float(rho),
                "Spearman_p": float(p_s),
            }

            if motion_col in sub.columns and sub[motion_col].notna().sum() == len(sub):
                X = np.column_stack(
                    [
                        np.ones(len(sub)),
                        x,
                        sub[motion_col].to_numpy(dtype=float),
                    ]
                )
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                resid = y - X @ beta
                dof = len(y) - X.shape[1]
                if dof > 0:
                    s2 = float((resid @ resid) / dof)
                    cov = s2 * np.linalg.inv(X.T @ X)
                    se = np.sqrt(np.diag(cov))
                    tvals = beta / se
                    pvals = 2 * stats.t.sf(np.abs(tvals), dof)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                    r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
                    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / dof if np.isfinite(r2) else np.nan
                    row.update(
                        {
                            "Beta_predictor": float(beta[1]),
                            "SE_predictor": float(se[1]),
                            "t_predictor": float(tvals[1]),
                            "p_predictor": float(pvals[1]),
                            "Beta_motion": float(beta[2]),
                            "p_motion": float(pvals[2]),
                            "Adj_R2": float(adj_r2),
                        }
                    )

            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["Pearson_q"] = fdr_bh(out["Pearson_p"].to_numpy())
        if "p_predictor" in out.columns:
            mask = out["p_predictor"].notna()
            out.loc[mask, "p_predictor_q"] = fdr_bh(out.loc[mask, "p_predictor"].to_numpy())
    return out


def load_charm_panel():
    acute = pd.read_csv(output_path("attractor_concentration", "acute_delta_attractor_concentration.csv"))
    acute = acute[acute["Dataset"] == "CHARM"].copy()

    join = pd.read_csv(output_path("missing_meq_panels", "charm_current_metric_meq_join.csv"))
    keep = [
        "SubjectID",
        "SharedDelta_State1_Occupancy",
        "SharedDelta_State3_Occupancy",
        "Delta_MeanFD",
        "Delta_Mystical",
        "Delta_Transcendence",
        "Delta_MEQ_mean4",
    ]
    join = join[keep].copy()

    panel = acute.merge(join, on="SubjectID", how="inner", validate="one_to_one")
    panel = panel.rename(
        columns={
            "SharedDelta_State1_Occupancy": "Delta_State1_Occupancy",
            "SharedDelta_State3_Occupancy": "Delta_State3_Occupancy",
            "Delta_MEQ_mean4": "Outcome_Overall",
            "Delta_Mystical": "Outcome_Mystical",
            "Delta_Transcendence": "Outcome_Transcendence",
        }
    )
    return panel


def load_psiconnect_panel():
    acute = pd.read_csv(output_path("attractor_concentration", "acute_delta_attractor_concentration.csv"))
    acute = acute[acute["Dataset"] == "PsiConnect"].copy()

    join = pd.read_csv(output_path("missing_meq_panels", "psiconnect_sharedstate_meq_join.csv"))
    keep = [
        "SubjectID",
        "Delta_State1_Occupancy",
        "Delta_State3_Occupancy",
        "Delta_MeanFD",
        "MEQ30_MEAN",
        "MEQ30_MYSTICAL",
        "MEQ30_TRANSCEND",
    ]
    join = join[keep].copy()

    panel = acute.merge(join, on="SubjectID", how="inner", validate="one_to_one", suffixes=("", "_join"))
    if "Delta_MeanFD_join" in panel.columns:
        panel["Delta_MeanFD"] = panel["Delta_MeanFD_join"]
        panel = panel.drop(columns=["Delta_MeanFD_join"])
    panel = panel.rename(
        columns={
            "MEQ30_MEAN": "Outcome_Overall",
            "MEQ30_MYSTICAL": "Outcome_Mystical",
            "MEQ30_TRANSCEND": "Outcome_Transcendence",
        }
    )
    return panel


def common_predictors():
    return [
        "DeltaQED",
        "DeltaSwitchRate",
        "Delta_DominantStateOccupancy",
        "Delta_OccupancyEntropy_bits",
        "Delta_WeightedStateDispersion",
        "Delta_State1_Occupancy",
        "Delta_State3_Occupancy",
    ]


def common_outcomes():
    return [
        "Outcome_Overall",
        "Outcome_Mystical",
        "Outcome_Transcendence",
    ]


def top_summary(df):
    cols = [
        "Predictor",
        "Outcome",
        "N",
        "Pearson_r",
        "Pearson_p",
        "Pearson_q",
        "Beta_predictor",
        "p_predictor",
        "p_predictor_q",
        "Adj_R2",
    ]
    keep = [c for c in cols if c in df.columns]
    return df.sort_values(["Pearson_p", "p_predictor" if "p_predictor" in df.columns else "Pearson_p"])[keep]


def sign_concordance(charm_res, psi_res):
    merged = charm_res.merge(
        psi_res,
        on=["Predictor", "Outcome"],
        suffixes=("_CHARM", "_PsiConnect"),
        how="inner",
    )
    merged["SameSign"] = np.sign(merged["Pearson_r_CHARM"]) == np.sign(merged["Pearson_r_PsiConnect"])
    return merged.sort_values(["SameSign", "Pearson_p_PsiConnect", "Pearson_p_CHARM"], ascending=[False, True, True])


def main():
    predictors = common_predictors()
    outcomes = common_outcomes()

    charm = load_charm_panel()
    charm.to_csv(OUTDIR / "charm_restricted_join.csv", index=False)
    charm_res = raw_and_adjusted(charm, predictors, outcomes, "Delta_MeanFD", "CHARM_restricted")
    charm_res.to_csv(OUTDIR / "charm_restricted_results.csv", index=False)
    top_summary(charm_res).to_csv(OUTDIR / "charm_restricted_top_results.csv", index=False)

    psi = load_psiconnect_panel()
    psi.to_csv(OUTDIR / "psiconnect_restricted_join.csv", index=False)
    psi_res = raw_and_adjusted(psi, predictors, outcomes, "Delta_MeanFD", "PsiConnect_restricted")
    psi_res.to_csv(OUTDIR / "psiconnect_restricted_results.csv", index=False)
    top_summary(psi_res).to_csv(OUTDIR / "psiconnect_restricted_top_results.csv", index=False)

    concord = sign_concordance(charm_res, psi_res)
    concord.to_csv(OUTDIR / "charm_psiconnect_sign_concordance.csv", index=False)


if __name__ == "__main__":
    main()
