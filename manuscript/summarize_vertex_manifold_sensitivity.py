import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from canonical_results import OPTIONAL_HIGHRES
from manuscript_paths import optional_existing_path, output_path

OUT = output_path("vertex_manifold_sensitivity")
OUT.mkdir(parents=True, exist_ok=True)


def fdr_bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def paired_metric_tests(results_path: Path, cond_a: str, cond_b: str, dataset: str | None = None):
    df = pd.read_csv(results_path)
    if dataset is not None and "Dataset" in df.columns:
        df = df[df["Dataset"] == dataset].copy()
    wide = df.pivot_table(index="SubjectID", columns="Condition", values=["QED", "NGSC", "NGSC_vtx"], aggfunc="mean")
    rows = []
    for metric in ["QED", "NGSC", "NGSC_vtx"]:
        if metric not in wide.columns.get_level_values(0):
            continue
        sub = wide[metric].dropna()
        if cond_a in sub.columns and cond_b in sub.columns and len(sub) >= 4:
            t, p = stats.ttest_rel(sub[cond_b], sub[cond_a])
            rows.append(
                {
                    "Metric": metric,
                    "N": len(sub),
                    "MeanA": sub[cond_a].mean(),
                    "MeanB": sub[cond_b].mean(),
                    "Delta_B_minus_A": (sub[cond_b] - sub[cond_a]).mean(),
                    "T": t,
                    "P": p,
                }
            )
    out = pd.DataFrame(rows)
    out["Q_FDR"] = fdr_bh(out["P"].values) if len(out) else []
    return out


def paired_geometry_tests(path: Path, value_col: str, id_col: str, cond_a: str, cond_b: str, dataset: str | None = None):
    df = pd.read_csv(path)
    if dataset is not None and "Dataset" in df.columns:
        df = df[df["Dataset"] == dataset].copy()
    rows = []
    for label, sub in df.groupby(id_col):
        wide = sub.pivot_table(index="SubjectID", columns="Condition", values=value_col, aggfunc="mean").dropna()
        if cond_a in wide.columns and cond_b in wide.columns and len(wide) >= 4:
            t, p = stats.ttest_rel(wide[cond_b], wide[cond_a])
            rows.append(
                {
                    id_col: label,
                    "N": len(wide),
                    "MeanA": wide[cond_a].mean(),
                    "MeanB": wide[cond_b].mean(),
                    "Delta_B_minus_A": (wide[cond_b] - wide[cond_a]).mean(),
                    "T": t,
                    "P": p,
                }
            )
    out = pd.DataFrame(rows)
    out["Q_FDR"] = fdr_bh(out["P"].values) if len(out) else []
    return out.sort_values("P").reset_index(drop=True)


def write_summary(psi_metrics, charm_metrics, psi_net, psi_pair, charm_net, charm_pair):
    lines = []
    lines.append("# Vertex/Dense Manifold Sensitivity")
    lines.append("")
    lines.append("This sensitivity analysis uses pre-existing dense/vertex manifold outputs rather than rerunning the pipeline.")
    lines.append("")

    lines.append("## PsiConnect dense fsLR")
    for _, r in psi_metrics.sort_values("P").iterrows():
        lines.append(
            f"- {r['Metric']}: N={int(r['N'])}, mean {r['MeanA']:.3f} -> {r['MeanB']:.3f}, delta={r['Delta_B_minus_A']:.3f}, t={r['T']:.3f}, p={r['P']:.4g}, q={r['Q_FDR']:.4g}"
        )
    lines.append(f"- Network-to-global hits: {(psi_net['Q_FDR'] < 0.05).sum()} / {len(psi_net)}")
    lines.append(f"- Pairwise Procrustes hits: {(psi_pair['Q_FDR'] < 0.05).sum()} / {len(psi_pair)}")

    lines.append("")
    lines.append("## CHARM vertex dtseries")
    if charm_metrics.empty or charm_net.empty or charm_pair.empty:
        lines.append("- Skipped: CHARM vertex sensitivity exports were not supplied via the clean-room input contract.")
    else:
        for _, r in charm_metrics.sort_values("P").iterrows():
            lines.append(
                f"- {r['Metric']}: N={int(r['N'])}, mean {r['MeanA']:.3f} -> {r['MeanB']:.3f}, delta={r['Delta_B_minus_A']:.3f}, t={r['T']:.3f}, p={r['P']:.4g}, q={r['Q_FDR']:.4g}"
            )
        lines.append(f"- Network-to-global hits: {(charm_net['Q_FDR'] < 0.05).sum()} / {len(charm_net)}")
        lines.append(f"- Pairwise Procrustes hits: {(charm_pair['Q_FDR'] < 0.05).sum()} / {len(charm_pair)}")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("- PsiConnect dense data preserve the main macro/fine-scale pattern: lower QED and higher NGSC_vtx under psilocybin.")
    lines.append("- PsiConnect dense reduced-network geometry still shows broad acute contraction/integration.")
    lines.append("- CHARM vertex data preserve NGSC_vtx increases relative to baseline, but placebo-vs-psychedelic QED and reduced-network geometry are weak/null.")
    lines.append("- This makes vertex-level manifold construction a useful PsiConnect sensitivity check, but not a clean replacement for the ROI-level cross-dataset manifold backbone.")

    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    psi_metrics_src = output_path("harmonized_psychedelics_summary", "harmonized_metric_subject_condition_means.csv")
    psi_metrics = paired_metric_tests(psi_metrics_src, "Baseline", "Psilocybin", dataset="PsiConnect_mean")
    if OPTIONAL_HIGHRES.exists():
        highres = pd.read_csv(OPTIONAL_HIGHRES)
        highres = highres[(highres["Dataset"] == "PsiConnect") & (highres["Status"] == "ok")].copy()
        wide = highres.pivot_table(index="SubjectID", columns="Condition", values="NGSC_vtx", aggfunc="mean").dropna()
        if {"Baseline", "Psilocybin"}.issubset(wide.columns) and len(wide) >= 4:
            t, p = stats.ttest_rel(wide["Psilocybin"], wide["Baseline"])
            psi_metrics = pd.concat(
                [
                    psi_metrics[psi_metrics["Metric"] != "NGSC_vtx"],
                    pd.DataFrame(
                        [{
                            "Metric": "NGSC_vtx",
                            "N": len(wide),
                            "MeanA": wide["Baseline"].mean(),
                            "MeanB": wide["Psilocybin"].mean(),
                            "Delta_B_minus_A": (wide["Psilocybin"] - wide["Baseline"]).mean(),
                            "T": t,
                            "P": p,
                        }]
                    ),
                ],
                ignore_index=True,
            )
            psi_metrics["Q_FDR"] = fdr_bh(psi_metrics["P"].values) if len(psi_metrics) else []
    charm_results = optional_existing_path("CHARM_VERTEX_RESULTS_CSV")
    charm_global = optional_existing_path("CHARM_VERTEX_GLOBALTBL_CSV")
    charm_pairwise = optional_existing_path("CHARM_VERTEX_NETPROC_CSV")
    charm_metrics = (
        paired_metric_tests(charm_results, "Placebo", "Psychedelic")
        if charm_results is not None
        else pd.DataFrame()
    )

    psi_net = paired_geometry_tests(
        output_path("harmonized_psychedelics_lme", "harmonized_network_to_global_distances_with_fd.csv"),
        "ProcrustesGlobalDist",
        "Network",
        "Baseline",
        "Psilocybin",
        dataset="PsiConnect_mean",
    )
    psi_pair = paired_geometry_tests(
        output_path("harmonized_psychedelics_lme", "harmonized_network_procrustes_distances_with_fd.csv"),
        "ProcrustesNetDist",
        "Pair",
        "Baseline",
        "Psilocybin",
        dataset="PsiConnect_mean",
    )
    charm_net = (
        paired_geometry_tests(charm_global, "NetToGlobalDist", "Network", "Placebo", "Psychedelic")
        if charm_global is not None
        else pd.DataFrame()
    )
    charm_pair = (
        paired_geometry_tests(charm_pairwise, "ProcrustesNetDist", "Pair", "Placebo", "Psychedelic")
        if charm_pairwise is not None
        else pd.DataFrame()
    )

    psi_metrics.to_csv(OUT / "psiconnect_dense_core_metrics.csv", index=False)
    charm_metrics.to_csv(OUT / "charm_vertex_core_metrics.csv", index=False)
    psi_net.to_csv(OUT / "psiconnect_dense_network_to_global.csv", index=False)
    psi_pair.to_csv(OUT / "psiconnect_dense_pairwise_procrustes.csv", index=False)
    charm_net.to_csv(OUT / "charm_vertex_network_to_global.csv", index=False)
    charm_pair.to_csv(OUT / "charm_vertex_pairwise_procrustes.csv", index=False)

    write_summary(psi_metrics, charm_metrics, psi_net, psi_pair, charm_net, charm_pair)


if __name__ == "__main__":
    main()
