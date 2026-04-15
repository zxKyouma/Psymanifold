from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from canonical_results import (
    core_subject_condition,
    state_centroids,
    state_subject_condition,
)
from manuscript_paths import output_path
OUTDIR = output_path("attractor_concentration")
OUTDIR.mkdir(parents=True, exist_ok=True)

LEIDA_K = 6
STATE_OCC_OUT = [f"State{s}_Occupancy" for s in range(1, LEIDA_K + 1)]
STATE_COLS = STATE_OCC_OUT
CONC_METRICS = [
    "DominantStateOccupancy",
    "Top1Top2Gap",
    "OccupancyEntropy_bits",
    "EffectiveNumberOfStates",
    "WeightedStateDispersion",
]
TARGET_METRICS = ["DeltaQED", "DeltaSwitchRate", "DeltaEntropy"]


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
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all():
        return np.nan
    total = p.sum()
    if total <= 0:
        return np.nan
    p = p / total
    mu = p @ centroids
    sq = ((centroids - mu) ** 2).sum(axis=1)
    return float(np.sqrt((p * sq).sum()))


def compute_concentration_metrics(df: pd.DataFrame, centroids: np.ndarray) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        p = row[STATE_COLS].to_numpy(dtype=float)
        top2 = np.sort(p)[-2:]
        ent = occupancy_entropy_bits(p)
        rows.append(
            {
                **row.to_dict(),
                "DominantStateOccupancy": float(np.max(p)),
                "Top1Top2Gap": float(top2[-1] - top2[-2]),
                "OccupancyEntropy_bits": ent,
                "EffectiveNumberOfStates": float(2 ** ent) if np.isfinite(ent) else np.nan,
                "WeightedStateDispersion": weighted_state_dispersion(p, centroids),
            }
        )
    return pd.DataFrame(rows)


def build_psiconnect_subject_condition() -> pd.DataFrame:
    df = state_subject_condition("PsiConnect_mean")
    out = df[["SubjectID", "Condition", *STATE_COLS]].copy()
    out.insert(0, "DrugClass", "PsilocybinClass")
    out.insert(0, "Dataset", "PsiConnect")
    out = out.rename(columns={f"Occupancy_State{s}": f"State{s}_Occupancy" for s in range(1, LEIDA_K + 1)})
    return out


def build_shared_state_subject_condition(dataset: str, drug_class: str) -> pd.DataFrame:
    df = state_subject_condition(dataset)
    out = df[["SubjectID", "Condition", *STATE_COLS]].copy()
    out.insert(0, "DrugClass", drug_class)
    out.insert(0, "Dataset", dataset if dataset != "CHARM_Psilocybin" else "CHARM")
    out = out.rename(columns={f"Occupancy_State{s}": f"State{s}_Occupancy" for s in range(1, LEIDA_K + 1)})
    return out


def build_subject_condition_panel() -> pd.DataFrame:
    datasets = []

    psi_occ = build_psiconnect_subject_condition()
    psi_centroids = state_centroids("PsiConnect_mean")
    datasets.append(compute_concentration_metrics(psi_occ, psi_centroids))

    charm_occ = build_shared_state_subject_condition("CHARM_Psilocybin", "PsilocybinClass")
    charm_centroids = state_centroids("CHARM_Psilocybin")
    datasets.append(compute_concentration_metrics(charm_occ, charm_centroids))

    lsd_occ = build_shared_state_subject_condition("LSD", "LSD")
    lsd_centroids = state_centroids("LSD")
    datasets.append(compute_concentration_metrics(lsd_occ, lsd_centroids))

    conc_df = pd.concat(datasets, ignore_index=True)

    core = pd.concat(
        [
            core_subject_condition("PsiConnect_mean").assign(Dataset="PsiConnect"),
            core_subject_condition("CHARM_Psilocybin").assign(Dataset="CHARM"),
            core_subject_condition("LSD").assign(Dataset="LSD"),
        ],
        ignore_index=True,
    )[["Dataset", "SubjectID", "Condition", "MeanFD", "QED", "SwitchRate", "Entropy"]]
    panel = conc_df.merge(core, on=["Dataset", "SubjectID", "Condition"], how="left", validate="one_to_one")
    return panel.sort_values(["Dataset", "SubjectID", "Condition"]).reset_index(drop=True)


def acute_condition_map(dataset: str) -> tuple[str, str]:
    if dataset == "PsiConnect":
        return "Baseline", "Psilocybin"
    if dataset == "CHARM":
        return "Placebo", "Psychedelic"
    if dataset == "LSD":
        return "Placebo", "Psychedelic"
    raise ValueError(dataset)


def paired_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 3:
        return np.nan, np.nan
    tval, pval = stats.ttest_rel(a, b, nan_policy="omit")
    return float(tval), float(pval)


def pearsonr_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(x, y)
    return float(r), float(p), n


def ols_binary_group(delta: np.ndarray, is_lsd: np.ndarray) -> tuple[float, float, float, int]:
    mask = np.isfinite(delta) & np.isfinite(is_lsd)
    y = delta[mask].astype(float)
    g = is_lsd[mask].astype(float)
    n = len(y)
    if n < 4 or len(np.unique(g)) < 2:
        return np.nan, np.nan, np.nan, n
    X = np.column_stack([np.ones(n), g])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    df = n - X.shape[1]
    if df <= 0:
        return float(beta[1]), np.nan, np.nan, n
    sigma2 = float((resid @ resid) / df)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = float(np.sqrt(cov[1, 1]))
    tval = float(beta[1] / se) if se > 0 else np.nan
    pval = float(2 * stats.t.sf(abs(tval), df)) if np.isfinite(tval) else np.nan
    return float(beta[1]), tval, pval, n


def build_delta_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, dsub in panel.groupby("Dataset"):
        cond_a, cond_b = acute_condition_map(dataset)
        a = dsub[dsub["Condition"] == cond_a].set_index("SubjectID")
        b = dsub[dsub["Condition"] == cond_b].set_index("SubjectID")
        common = a.index.intersection(b.index)
        for sid in common:
            ra = a.loc[sid]
            rb = b.loc[sid]
            rec = {
                "Dataset": dataset,
                "DrugClass": "LSD" if dataset == "LSD" else "PsilocybinClass",
                "SubjectID": sid,
                "ConditionA": cond_a,
                "ConditionB": cond_b,
            }
            for metric in CONC_METRICS:
                rec[f"{cond_a}_{metric}"] = float(ra[metric])
                rec[f"{cond_b}_{metric}"] = float(rb[metric])
                rec[f"Delta_{metric}"] = float(rb[metric] - ra[metric])
            for metric, out_name in [("QED", "DeltaQED"), ("SwitchRate", "DeltaSwitchRate"), ("Entropy", "DeltaEntropy")]:
                rec[f"{cond_a}_{metric}"] = float(ra[metric])
                rec[f"{cond_b}_{metric}"] = float(rb[metric])
                rec[out_name] = float(rb[metric] - ra[metric])
            rec["DeltaMeanFD"] = float(rb["MeanFD"] - ra["MeanFD"]) if np.isfinite(ra["MeanFD"]) and np.isfinite(rb["MeanFD"]) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(["Dataset", "SubjectID"]).reset_index(drop=True)


def acute_paired_tests(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, dsub in panel.groupby("Dataset"):
        cond_a, cond_b = acute_condition_map(dataset)
        a = dsub[dsub["Condition"] == cond_a].set_index("SubjectID")
        b = dsub[dsub["Condition"] == cond_b].set_index("SubjectID")
        common = a.index.intersection(b.index)
        for metric in CONC_METRICS:
            va = a.loc[common, metric].to_numpy(dtype=float)
            vb = b.loc[common, metric].to_numpy(dtype=float)
            tval, pval = paired_t(va, vb)
            rows.append(
                {
                    "Dataset": dataset,
                    "DrugClass": "LSD" if dataset == "LSD" else "PsilocybinClass",
                    "Contrast": "acute",
                    "ConditionA": cond_a,
                    "ConditionB": cond_b,
                    "Metric": metric,
                    "N": len(common),
                    "MeanA": float(np.nanmean(va)),
                    "MeanB": float(np.nanmean(vb)),
                    "Delta_B_minus_A": float(np.nanmean(vb - va)),
                    "T": tval,
                    "P": pval,
                }
            )
    out = pd.DataFrame(rows)
    out["Q_FDR"] = bh_fdr(out["P"])
    return out


def drugclass_delta_tests(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in CONC_METRICS:
        col = f"Delta_{metric}"
        psilo = delta_df.loc[delta_df["DrugClass"] == "PsilocybinClass", col].to_numpy(dtype=float)
        lsd = delta_df.loc[delta_df["DrugClass"] == "LSD", col].to_numpy(dtype=float)
        beta, tval, pval, n = ols_binary_group(delta_df[col].to_numpy(dtype=float), (delta_df["DrugClass"] == "LSD").astype(int).to_numpy())
        rows.append(
            {
                "Metric": metric,
                "N_total": n,
                "N_psilocybin_class": int(np.isfinite(psilo).sum()),
                "N_lsd": int(np.isfinite(lsd).sum()),
                "MeanDelta_psilocybin_class": float(np.nanmean(psilo)),
                "MeanDelta_lsd": float(np.nanmean(lsd)),
                "LSD_minus_PsilocybinClass_Beta": beta,
                "T": tval,
                "P": pval,
            }
        )
    out = pd.DataFrame(rows)
    out["Q_FDR"] = bh_fdr(out["P"])
    return out


def coupling_tests(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = {
        "All": delta_df,
        "PsilocybinClass": delta_df[delta_df["DrugClass"] == "PsilocybinClass"],
        "PsiConnect": delta_df[delta_df["Dataset"] == "PsiConnect"],
        "CHARM": delta_df[delta_df["Dataset"] == "CHARM"],
        "LSD": delta_df[delta_df["Dataset"] == "LSD"],
    }
    for scope, sub in scopes.items():
        for metric in CONC_METRICS:
            x = sub[f"Delta_{metric}"].to_numpy(dtype=float)
            for target in TARGET_METRICS:
                y = sub[target].to_numpy(dtype=float)
                r, p, n = pearsonr_safe(x, y)
                rows.append(
                    {
                        "Scope": scope,
                        "Metric": metric,
                        "Target": target,
                        "N": n,
                        "R": r,
                        "R2": float(r * r) if np.isfinite(r) else np.nan,
                        "P": p,
                    }
                )
    out = pd.DataFrame(rows)
    out["Q_FDR"] = bh_fdr(out["P"])
    return out


def main() -> None:
    panel = build_subject_condition_panel()
    panel.to_csv(OUTDIR / "subject_condition_attractor_concentration.csv", index=False)

    delta_df = build_delta_table(panel)
    delta_df.to_csv(OUTDIR / "acute_delta_attractor_concentration.csv", index=False)

    paired = acute_paired_tests(panel)
    paired.to_csv(OUTDIR / "acute_paired_tests.csv", index=False)

    interactions = drugclass_delta_tests(delta_df)
    interactions.to_csv(OUTDIR / "drugclass_delta_interactions.csv", index=False)

    couplings = coupling_tests(delta_df)
    couplings.to_csv(OUTDIR / "coupling_with_qed_switch_entropy.csv", index=False)

    summary_rows = []
    for metric in CONC_METRICS:
        pair = paired[paired["Metric"] == metric]
        inter = interactions[interactions["Metric"] == metric]
        summary_rows.append(
            {
                "Metric": metric,
                "PsiConnect_Delta": float(pair.loc[pair["Dataset"] == "PsiConnect", "Delta_B_minus_A"].iloc[0]),
                "PsiConnect_P": float(pair.loc[pair["Dataset"] == "PsiConnect", "P"].iloc[0]),
                "CHARM_Delta": float(pair.loc[pair["Dataset"] == "CHARM", "Delta_B_minus_A"].iloc[0]),
                "CHARM_P": float(pair.loc[pair["Dataset"] == "CHARM", "P"].iloc[0]),
                "LSD_Delta": float(pair.loc[pair["Dataset"] == "LSD", "Delta_B_minus_A"].iloc[0]),
                "LSD_P": float(pair.loc[pair["Dataset"] == "LSD", "P"].iloc[0]),
                "DrugClass_LSD_minus_Psilo_Beta": float(inter["LSD_minus_PsilocybinClass_Beta"].iloc[0]),
                "DrugClass_P": float(inter["P"].iloc[0]),
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUTDIR / "summary_key_results.csv", index=False)


if __name__ == "__main__":
    main()
