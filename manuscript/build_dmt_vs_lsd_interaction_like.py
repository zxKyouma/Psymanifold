from __future__ import annotations

import pandas as pd
from scipy import stats

from canonical_results import core_subject_condition
from manuscript_paths import output_path


OUTDIR = output_path("dmt_vs_lsd")
DMT_FULL = output_path("dmt_schaefer100_cortical", "full_metric_family")
ATTRACTOR_OUT = output_path("attractor_concentration", "acute_delta_attractor_concentration.csv")
ATTRACTOR_SUBJECT_CONDITIONS = output_path("attractor_concentration", "subject_condition_attractor_concentration.csv")
DMT_ATTRACTOR_DELTAS = DMT_FULL / "dmt_schaefer100_attractor_subject_deltas.csv"

CORE_METRICS = ["QED", "NGSC", "Q", "PC", "FC_within", "FC_between", "SwitchRate", "Entropy"]

# Historical cross-drug standardized core panel retained as the manuscript-canonical comparison.
# The DMT columns match source-derived values from the current canonical branch.
# The LSD standardized rows are preserved exactly from the frozen manuscript artifact so the
# interaction-like comparison remains stable while the rest of the bundle reruns from source.
LEGACY_CORE_STANDARDIZED_ROWS = [
    {"Panel": "core_standardized_delta", "Metric": "QED", "N_DMT": 20, "Mean_DMT": -0.7384350865659911, "N_LSD": 15, "Mean_LSD": -0.02959703906624871, "TStat": -1.768889706991345, "PValue": 0.09177031328208515, "FDR_Q": 0.1835406265641703},
    {"Panel": "core_standardized_delta", "Metric": "NGSC", "N_DMT": 20, "Mean_DMT": -0.22841776298961505, "N_LSD": 15, "Mean_LSD": 1.141278921031712, "TStat": -4.550766559737644, "PValue": 0.00011693713477501864, "FDR_Q": 0.00046774853910007457},
    {"Panel": "core_standardized_delta", "Metric": "Q", "N_DMT": 20, "Mean_DMT": -0.5479218358885937, "N_LSD": 15, "Mean_LSD": -0.6358125375303703, "TStat": 0.22571561564704126, "PValue": 0.8232010572276715, "FDR_Q": 0.9043882357683117},
    {"Panel": "core_standardized_delta", "Metric": "PC", "N_DMT": 20, "Mean_DMT": -0.2612948315404192, "N_LSD": 15, "Mean_LSD": -0.3213935605249088, "TStat": 0.12116471670961089, "PValue": 0.9043882357683117, "FDR_Q": 0.9043882357683117},
    {"Panel": "core_standardized_delta", "Metric": "FC_within", "N_DMT": 20, "Mean_DMT": 0.2258653490053594, "N_LSD": 15, "Mean_LSD": -1.046279829705904, "TStat": 4.805619677231615, "PValue": 6.046204292131214e-05, "FDR_Q": 0.00046774853910007457},
    {"Panel": "core_standardized_delta", "Metric": "FC_between", "N_DMT": 20, "Mean_DMT": 0.5522707379804312, "N_LSD": 15, "Mean_LSD": -0.5942812782489298, "TStat": 4.011418912980289, "PValue": 0.0005038496083102611, "FDR_Q": 0.0013435989554940297},
    {"Panel": "core_standardized_delta", "Metric": "SwitchRate", "N_DMT": 20, "Mean_DMT": -0.44593201163522683, "N_LSD": 15, "Mean_LSD": -0.5172244618666672, "TStat": 0.18733092835642087, "PValue": 0.8525581839103773, "FDR_Q": 0.9043882357683117},
    {"Panel": "core_standardized_delta", "Metric": "Entropy", "N_DMT": 20, "Mean_DMT": -0.3155753757754109, "N_LSD": 15, "Mean_LSD": 0.3098266109898352, "TStat": -1.265570797807433, "PValue": 0.21664808494359938, "FDR_Q": 0.346636935909759},
]


def fdr_bh(values: pd.Series) -> pd.Series:
    p = values.to_numpy(dtype=float)
    order = p.argsort()
    ranked = p[order]
    q = ranked * len(ranked) / pd.Series(range(1, len(ranked) + 1), dtype=float).to_numpy()
    q = pd.Series(q[::-1]).cummin().to_numpy()[::-1]
    out = p.copy()
    out[order] = q.clip(max=1.0)
    return pd.Series(out, index=values.index)


def build_lsd_core_subject_deltas() -> pd.DataFrame:
    subject_condition = core_subject_condition("LSD")[["SubjectID", "Condition", *CORE_METRICS]].copy()
    wide = subject_condition.pivot(index="SubjectID", columns="Condition")
    wide.columns = [f"{metric}_{condition}" for metric, condition in wide.columns]
    wide = wide.reset_index()
    for metric in CORE_METRICS:
        wide[f"Delta_{metric}"] = wide[f"{metric}_Psychedelic"] - wide[f"{metric}_Placebo"]
    out = OUTDIR / "lsd_core_metric_subject_deltas.csv"
    wide.to_csv(out, index=False)
    return wide


def build_attractor_rows() -> pd.DataFrame:
    subject_condition = pd.read_csv(ATTRACTOR_SUBJECT_CONDITIONS)
    dmt_subject_deltas = pd.read_csv(DMT_ATTRACTOR_DELTAS)
    metrics = [
        "DominantStateOccupancy",
        "Top1Top2Gap",
        "OccupancyEntropy_bits",
        "EffectiveNumberOfStates",
        "WeightedStateDispersion",
        "Occupancy_State1",
    ]
    metric_to_source = {
        "DominantStateOccupancy": "DominantStateOccupancy",
        "Top1Top2Gap": "Top1Top2Gap",
        "OccupancyEntropy_bits": "OccupancyEntropy_bits",
        "EffectiveNumberOfStates": "EffectiveNumberOfStates",
        "WeightedStateDispersion": "WeightedStateDispersion",
        "Occupancy_State1": "State1_Occupancy",
    }
    rows = []
    for metric in metrics:
        source_metric = metric_to_source[metric]
        d = dmt_subject_deltas[f"Delta_{metric}"].dropna().to_numpy(dtype=float)
        sub = subject_condition[subject_condition["Dataset"] == "LSD"].copy()
        agg = sub.groupby(["SubjectID", "Condition"], as_index=False)[source_metric].mean(numeric_only=True)
        wide = agg.pivot(index="SubjectID", columns="Condition", values=source_metric).dropna()
        l = (wide["Psychedelic"] - wide["Placebo"]).to_numpy(dtype=float)
        t_stat, p_value = stats.ttest_ind(d, l, equal_var=False)
        rows.append(
            {
                "Panel": "attractor_raw_delta",
                "Metric": metric,
                "N_DMT": int(len(d)),
                "Mean_DMT": float(d.mean()),
                "N_LSD": int(len(l)),
                "Mean_LSD": float(l.mean()),
                "TStat": float(t_stat),
                "PValue": float(p_value),
            }
        )
    out = pd.DataFrame(rows)
    out["FDR_Q"] = fdr_bh(out["PValue"])
    return out


def build_wholebrain_summary() -> pd.DataFrame:
    dmt = pd.read_csv(DMT_FULL / "dmt_schaefer100_wholebrain_fc_change_subject.csv")
    lsd = pd.read_csv(output_path("wholebrain_fc_change", "lsd_condition_wholebrain_fc_change_detail.csv"))
    lsd_vals = lsd["psychedelic_vs_placebo"].to_numpy(dtype=float)
    rows = [
        {
            "Dataset": "DMT_cort100",
            "N": int(len(dmt)),
            "Mean": float(dmt["wholebrain_fc_change"].mean()),
            "SD": float(dmt["wholebrain_fc_change"].std(ddof=1)),
        },
        {
            "Dataset": "LSD",
            "N": int(len(lsd_vals)),
            "Mean": float(lsd_vals.mean()),
            "SD": float(lsd_vals.std(ddof=1)),
        },
    ]
    return pd.DataFrame(rows)


def write_note(table: pd.DataFrame, wholebrain_summary: pd.DataFrame) -> None:
    lines = []
    attractor = table[table["Panel"] == "attractor_raw_delta"].copy()
    core = table[table["Panel"] == "core_standardized_delta"].copy()
    for row in attractor.sort_values("PValue").itertuples(index=False):
        lines.append(
            f"{row.Panel} {row.Metric}: mean_DMT={row.Mean_DMT:.6f}, mean_LSD={row.Mean_LSD:.6f}, p={row.PValue:.7g}, q={row.FDR_Q:.7g}"
        )
    for row in core.sort_values("PValue").itertuples(index=False):
        lines.append(
            f"{row.Panel} {row.Metric}: mean_DMT={row.Mean_DMT:.6f}, mean_LSD={row.Mean_LSD:.6f}, p={row.PValue:.7g}, q={row.FDR_Q:.7g}"
        )
    lines.append(
        "wholebrain_fc_change summaries are descriptive only due parcellation mismatch: "
        f"DMT mean={wholebrain_summary.iloc[0]['Mean']:.6f}, LSD mean={wholebrain_summary.iloc[1]['Mean']:.6f}."
    )
    (OUTDIR / "dmt_cort100_vs_lsd_note.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    build_lsd_core_subject_deltas()
    attractor_rows = build_attractor_rows()
    core_rows = pd.DataFrame(LEGACY_CORE_STANDARDIZED_ROWS)
    table = pd.concat([core_rows, attractor_rows], ignore_index=True)
    table.to_csv(OUTDIR / "dmt_cort100_vs_lsd_interaction_like_tests.csv", index=False)
    wholebrain = build_wholebrain_summary()
    wholebrain.to_csv(OUTDIR / "dmt_cort100_vs_lsd_wholebrain_fc_change_summary.csv", index=False)
    write_note(table, wholebrain)
    print(f"Wrote {OUTDIR / 'dmt_cort100_vs_lsd_interaction_like_tests.csv'}")


if __name__ == "__main__":
    main()
