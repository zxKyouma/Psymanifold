import numpy as np
import pandas as pd
from scipy import stats

from canonical_results import (
    charm_cross_metric_deltas,
    charm_motion_deltas,
    psiconnect_sharedstate_meq_join,
    state_subject_condition,
)
from manuscript_paths import output_path, repo_path, require_path

OUTROOT = output_path("missing_meq_panels")


def fdr_bh(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        adj[i] = val
        prev = val
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out


def raw_and_adjusted(df, predictors, outcomes, motion_col, label):
    rows = []
    for predictor in predictors:
        for outcome in outcomes:
            cols = [predictor, outcome]
            if motion_col is not None and motion_col in df.columns:
                cols.append(motion_col)
            sub = df[cols].dropna().copy()
            if len(sub) < 4:
                continue

            r, p = stats.pearsonr(sub[predictor], sub[outcome])
            rho, p_s = stats.spearmanr(sub[predictor], sub[outcome])
            row = {
                "Panel": label,
                "Predictor": predictor,
                "Outcome": outcome,
                "N": len(sub),
                "Pearson_r": r,
                "Pearson_p": p,
                "Spearman_rho": rho,
                "Spearman_p": p_s,
            }

            if motion_col is not None and motion_col in sub.columns:
                X = np.column_stack(
                    [
                        np.ones(len(sub)),
                        sub[predictor].to_numpy(dtype=float),
                        sub[motion_col].to_numpy(dtype=float),
                    ]
                )
                y = sub[outcome].to_numpy(dtype=float)
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                resid = y - X @ beta
                n = len(y)
                p_cols = X.shape[1]
                dof = n - p_cols
                if dof <= 0:
                    continue
                s2 = (resid @ resid) / dof
                cov = s2 * np.linalg.inv(X.T @ X)
                se = np.sqrt(np.diag(cov))
                tvals = beta / se
                pvals = 2 * stats.t.sf(np.abs(tvals), dof)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
                adj_r2 = 1 - (1 - r2) * (n - 1) / dof if np.isfinite(r2) else np.nan
                row.update(
                    {
                        "Beta_predictor": beta[1],
                        "SE_predictor": se[1],
                        "t_predictor": tvals[1],
                        "p_predictor": pvals[1],
                        "Beta_motion": beta[2],
                        "p_motion": pvals[2],
                        "Adj_R2": adj_r2,
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
    metrics = charm_cross_metric_deltas()
    shared = state_subject_condition("CHARM_Psilocybin").rename(
        columns={f"Occupancy_State{s}": f"State{s}_Occupancy" for s in range(1, 7)}
    )
    acute = shared[shared["Condition"].isin(["Placebo", "Psychedelic"])].copy()
    wide = acute.pivot(index="SubjectID", columns="Condition")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for metric in [
        "SwitchRate",
        "Entropy",
        "State1_Occupancy",
        "State2_Occupancy",
        "State3_Occupancy",
        "State4_Occupancy",
        "State5_Occupancy",
        "State6_Occupancy",
    ]:
        wide[f"SharedDelta_{metric}"] = wide[f"{metric}_Psychedelic"] - wide[f"{metric}_Placebo"]

    old_wide = charm_motion_deltas()

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

    panel = metrics.merge(wide, on="SubjectID", how="left").merge(
        old_wide[["SubjectID", "Delta_MeanFD"]], on="SubjectID", how="left"
    )
    panel = panel.merge(
        meq_wide[
            [
                "SubjectID",
                "Delta_Mystical",
                "Delta_PositiveMood",
                "Delta_Transcendence",
                "Delta_Ineffability",
                "Delta_MEQ_mean4",
            ]
        ],
        on="SubjectID",
        how="left",
    )
    return panel


def load_psiconnect_panel():
    return psiconnect_sharedstate_meq_join()


def main():
    OUTROOT.mkdir(parents=True, exist_ok=True)

    charm = load_charm_panel()
    charm.to_csv(OUTROOT / "charm_current_metric_meq_join.csv", index=False)
    charm_predictors = [
        "Delta_QED",
        "Delta_NGSC",
        "Delta_NGSC_vtx",
        "Delta_Q",
        "Delta_FC_within",
        "Delta_FC_between",
        "Delta_Entropy",
        "Delta_SwitchRate",
        "SharedDelta_State1_Occupancy",
        "SharedDelta_State2_Occupancy",
        "SharedDelta_State3_Occupancy",
        "SharedDelta_State4_Occupancy",
        "SharedDelta_State5_Occupancy",
        "SharedDelta_State6_Occupancy",
        "SharedDelta_Entropy",
        "SharedDelta_SwitchRate",
    ]
    charm_outcomes = [
        "Delta_MEQ_mean4",
        "Delta_Mystical",
        "Delta_PositiveMood",
        "Delta_Transcendence",
        "Delta_Ineffability",
    ]
    charm_res = raw_and_adjusted(charm, charm_predictors, charm_outcomes, "Delta_MeanFD", "CHARM_current")
    charm_res.to_csv(OUTROOT / "charm_current_metric_meq_results.csv", index=False)

    psi = load_psiconnect_panel()
    psi.to_csv(OUTROOT / "psiconnect_sharedstate_meq_join.csv", index=False)
    psi_predictors = [
        "Delta_State1_Occupancy",
        "Delta_State2_Occupancy",
        "Delta_State3_Occupancy",
        "Delta_State4_Occupancy",
        "Delta_State5_Occupancy",
        "Delta_State6_Occupancy",
        "Delta_SwitchRate_group",
        "DeltaDispersion",
    ]
    psi_outcomes = [
        "MEQ30_MEAN",
        "MEQ30_MYSTICAL",
        "MEQ30_TRANSCEND",
        "MEQ30_POSITIVE",
        "MEQ30_INEFFABILITY",
    ]
    psi_res = raw_and_adjusted(psi, psi_predictors, psi_outcomes, "Delta_MeanFD", "PsiConnect_sharedstate")
    psi_res.to_csv(OUTROOT / "psiconnect_sharedstate_meq_results.csv", index=False)

    for name, df in [("charm", charm_res), ("psiconnect", psi_res)]:
        if df.empty:
            continue
        cols = ["Predictor", "Outcome", "N", "Pearson_r", "Pearson_p", "Pearson_q"]
        if "p_predictor" in df.columns:
            cols += ["Beta_predictor", "p_predictor", "p_predictor_q"]
        summary = df.sort_values(["Pearson_p", "p_predictor" if "p_predictor" in df.columns else "Pearson_p"]).reset_index(drop=True)
        summary[cols].to_csv(OUTROOT / f"{name}_top_results.csv", index=False)


if __name__ == "__main__":
    main()
