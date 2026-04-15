from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.io import loadmat

from manuscript_paths import configured_path, optional_existing_path, output_path, repo_path, require_path


LEIDA_K = 6
STATE_COLS = [f"Occupancy_State{s}" for s in range(1, LEIDA_K + 1)]

SUMMARY_METRICS = output_path("harmonized_psychedelics_summary", "harmonized_metric_subject_condition_means.csv")
STATE_RUN = output_path("harmonized_psychedelics_extended_metrics", "harmonized_state_dynamics_run_metrics.csv")
STATE_TEMPLATES = output_path("harmonized_psychedelics_extended_metrics", "harmonized_shared_state_templates.csv")
DISPERSION = output_path("harmonized_psychedelics_summary", "harmonized_centroid_dispersion_subject_condition_means.csv")
NET_GLOBAL = output_path("harmonized_psychedelics_lme", "harmonized_network_to_global_distances_with_fd.csv")
PAIRWISE = output_path("harmonized_psychedelics_lme", "harmonized_network_procrustes_distances_with_fd.csv")
ENERGY = output_path("harmonized_psychedelics_summary", "harmonized_energy_subject_condition_means.csv")
RESULTS_WITH_FD = output_path("harmonized_psychedelics_lme", "harmonized_results_summary_with_fd.csv")
OPTIONAL_HIGHRES = output_path("harmonized_psychedelics_optional", "optional_highres_ngsc.csv")
TSV_MEQ = "PSICONNECT_MEQ30_TSV"
TSV_ASC = "PSICONNECT_ASC11_TSV"


def core_subject_condition(dataset: str) -> pd.DataFrame:
    subj = pd.read_csv(SUMMARY_METRICS)
    subj = subj[subj["Dataset"] == dataset].copy()
    meanfd = pd.read_csv(RESULTS_WITH_FD)
    meanfd = (
        meanfd[meanfd["Dataset"] == dataset]
        .groupby(["SubjectID", "Condition"], as_index=False)["MeanFD_final"]
        .mean()
        .rename(columns={"MeanFD_final": "MeanFD"})
    )
    return subj.merge(meanfd, on=["SubjectID", "Condition"], how="left")


def state_subject_condition(dataset: str) -> pd.DataFrame:
    state = pd.read_csv(STATE_RUN)
    state = state[state["Dataset"] == dataset].copy()
    keep = ["MeanFD", "SwitchRate", "Entropy", *STATE_COLS]
    return (
        state.groupby(["SubjectID", "Condition"], as_index=False)
        .agg(**{col: (col, "mean") for col in keep})
    )


def state_centroids(dataset: str) -> np.ndarray:
    df = pd.read_csv(STATE_TEMPLATES)
    df = df[df["Dataset"] == dataset].copy()
    roi_cols = [c for c in df.columns if c.startswith("ROI_")]
    states = df["StateID"].astype(int).to_numpy()
    centroids = df[roi_cols].to_numpy(dtype=float)
    order = np.argsort(states)
    return centroids[order]


def optional_highres_deltas(dataset: str, cond_a: str, cond_b: str) -> pd.DataFrame:
    if not OPTIONAL_HIGHRES.exists():
        return pd.DataFrame(columns=["SubjectID", "Delta_NGSC_vtx"])
    df = pd.read_csv(OPTIONAL_HIGHRES)
    df = df[(df["Dataset"] == dataset) & (df["Status"] == "ok")].copy()
    if df.empty:
        return pd.DataFrame(columns=["SubjectID", "Delta_NGSC_vtx"])
    agg = df.groupby(["SubjectID", "Condition"], as_index=False)["NGSC_vtx"].mean()
    wide = agg.pivot(index="SubjectID", columns="Condition", values="NGSC_vtx")
    if cond_a not in wide.columns or cond_b not in wide.columns:
        return pd.DataFrame(columns=["SubjectID", "Delta_NGSC_vtx"])
    wide = wide.reset_index()
    wide["Delta_NGSC_vtx"] = wide[cond_b] - wide[cond_a]
    return wide[["SubjectID", "Delta_NGSC_vtx"]]


def psiconnect_meq() -> pd.DataFrame:
    return pd.read_csv(
        require_path(TSV_MEQ, "PsiConnect MEQ30 questionnaire TSV"),
        sep="\t",
    ).rename(columns={"participant_id": "SubjectID"})


def psiconnect_asc() -> pd.DataFrame:
    return pd.read_csv(
        require_path(TSV_ASC, "PsiConnect ASC11 questionnaire TSV"),
        sep="\t",
    ).rename(columns={"participant_id": "SubjectID"})


def psiconnect_core_meq_join() -> pd.DataFrame:
    wide = core_subject_condition("PsiConnect_mean").pivot(index="SubjectID", columns="Condition")
    wide.columns = [f"{metric}_{cond}" for metric, cond in wide.columns]
    wide = wide.reset_index()
    panel = pd.DataFrame({"SubjectID": wide["SubjectID"]})
    panel["Delta_MeanFD"] = wide["MeanFD_Psilocybin"] - wide["MeanFD_Baseline"]
    panel["Delta_QED"] = wide["QED_Psilocybin"] - wide["QED_Baseline"]
    panel["Delta_NGSC"] = wide["NGSC_Psilocybin"] - wide["NGSC_Baseline"]
    panel["Delta_FC_within"] = wide["FC_within_Psilocybin"] - wide["FC_within_Baseline"]

    pair = pd.read_csv(PAIRWISE)
    pair = pair[(pair["Dataset"] == "PsiConnect_mean") & (pair["Pair"] == "default-somatomotorHand")].copy()
    pair = pair.groupby(["SubjectID", "Condition"], as_index=False)["ProcrustesNetDist"].mean()
    pair_wide = pair.pivot(index="SubjectID", columns="Condition", values="ProcrustesNetDist").dropna().reset_index()
    pair_wide["Delta_DMN_SM"] = pair_wide["Psilocybin"] - pair_wide["Baseline"]
    panel = panel.merge(pair_wide[["SubjectID", "Delta_DMN_SM"]], on="SubjectID", how="left")
    panel = panel.merge(optional_highres_deltas("PsiConnect", "Baseline", "Psilocybin"), on="SubjectID", how="left")

    meq = psiconnect_meq()
    keep = [
        "SubjectID",
        "MEQ30_MEAN",
        "MEQ30_MYSTICAL",
        "MEQ30_POSITIVE",
        "MEQ30_TRANSCEND",
        "MEQ30_INEFFABILITY",
    ]
    return panel.merge(meq[keep], on="SubjectID", how="inner")


def psiconnect_sharedstate_meq_join() -> pd.DataFrame:
    state = state_subject_condition("PsiConnect_mean")
    wide = state.pivot(index="SubjectID", columns="Condition")
    wide.columns = [f"{metric}_{cond}" for metric, cond in wide.columns]
    wide = wide.reset_index()
    out = wide[["SubjectID"]].copy()
    for s in range(1, LEIDA_K + 1):
        a = f"Occupancy_State{s}_Baseline"
        b = f"Occupancy_State{s}_Psilocybin"
        out[f"Delta_State{s}_Occupancy"] = wide[b] - wide[a]
    out["Delta_SwitchRate_group"] = wide["SwitchRate_Psilocybin"] - wide["SwitchRate_Baseline"]
    out["Delta_MeanFD"] = wide["MeanFD_Psilocybin"] - wide["MeanFD_Baseline"]

    disp = pd.read_csv(DISPERSION)
    disp = disp[disp["Dataset"] == "PsiConnect_mean"][["SubjectID", "Condition", "RawDispersion"]]
    disp_wide = disp.pivot(index="SubjectID", columns="Condition", values="RawDispersion").dropna().reset_index()
    disp_wide["DeltaDispersion"] = disp_wide["Psilocybin"] - disp_wide["Baseline"]
    out = out.merge(disp_wide[["SubjectID", "DeltaDispersion"]], on="SubjectID", how="left")

    meq = psiconnect_meq()
    keep = [
        "SubjectID",
        "MEQ30_MEAN",
        "MEQ30_MYSTICAL",
        "MEQ30_POSITIVE",
        "MEQ30_TRANSCEND",
        "MEQ30_INEFFABILITY",
    ]
    return out.merge(meq[keep], on="SubjectID", how="left")


def charm_motion_deltas(cond_a: str = "Placebo", cond_b: str = "Psychedelic") -> pd.DataFrame:
    core = core_subject_condition("CHARM_Psilocybin")
    wide = core.pivot(index="SubjectID", columns="Condition", values="MeanFD").dropna().reset_index()
    if cond_a not in wide.columns or cond_b not in wide.columns:
        return pd.DataFrame(columns=["SubjectID", "Delta_MeanFD"])
    wide["Delta_MeanFD"] = wide[cond_b] - wide[cond_a]
    return wide[["SubjectID", "Delta_MeanFD"]]


def charm_cross_metric_deltas() -> pd.DataFrame:
    cross = pd.read_csv(output_path("harmonized_psychedelics_parallel_extensions", "harmonized_cross_metric_subject_deltas.csv"))
    cross = cross[cross["Dataset"] == "CHARM_Psilocybin"].copy()
    rename = {
        "QED": "Delta_QED",
        "NGSC": "Delta_NGSC",
        "FC_within": "Delta_FC_within",
        "FC_between": "Delta_FC_between",
        "SwitchRate": "Delta_SwitchRate",
        "Entropy": "Delta_Entropy",
        "RawDispersion": "Delta_RawDispersion",
        "Default_GlobalDist": "Delta_Default_GlobalDist",
        "DMN_SM_Procrustes": "Delta_DMN_SM_Procrustes",
    }
    cross = cross.rename(columns=rename)
    subj = core_subject_condition("CHARM_Psilocybin")
    qwide = subj.pivot(index="SubjectID", columns="Condition", values="Q").reset_index()
    qwide["Delta_Q"] = qwide["Psychedelic"] - qwide["Placebo"]
    out = cross.merge(qwide[["SubjectID", "Delta_Q"]], on="SubjectID", how="left")
    out["Delta_NGSC_vtx"] = np.nan
    charm_vtx = optional_existing_path("CHARM_VERTEX_RESULTS_CSV")
    if charm_vtx is not None:
        vtx = pd.read_csv(charm_vtx)
        if {"SubjectID", "Condition", "NGSC_vtx"}.issubset(vtx.columns):
            agg = vtx.groupby(["SubjectID", "Condition"], as_index=False)["NGSC_vtx"].mean()
            wide = agg.pivot(index="SubjectID", columns="Condition", values="NGSC_vtx").dropna().reset_index()
            if {"Placebo", "Psychedelic"}.issubset(wide.columns):
                wide["Delta_NGSC_vtx"] = wide["Psychedelic"] - wide["Placebo"]
                out = out.merge(wide[["SubjectID", "Delta_NGSC_vtx"]], on="SubjectID", how="left", suffixes=("", "_new"))
                if "Delta_NGSC_vtx_new" in out.columns:
                    out["Delta_NGSC_vtx"] = out["Delta_NGSC_vtx_new"]
                    out = out.drop(columns=["Delta_NGSC_vtx_new"])
    return out


def compute_fc_global_for_psiconnect() -> pd.DataFrame:
    mapping = pd.read_csv(repo_path("assets", "gordon_roi_network_mapping.csv"))
    valid_mask = mapping["Network"].astype(str).str.lower().ne("none").to_numpy()
    manifest = configured_path(
        "HARMONIZED_TS_MANIFEST",
        default_rel="outputs/harmonized_psychedelics/harmonized_timeseries_manifest.csv",
    )
    if manifest is None or not manifest.exists():
        manifest = configured_path(
            "HARMONIZED_TS_MANIFEST",
            default_rel="outputs/harmonized_psychedelics/harmonized_timeseries_manifest.tsv",
        )
    if manifest is None or not manifest.exists():
        raise FileNotFoundError(
            "Missing harmonized timeseries manifest. Run the public core pipeline first or set HARMONIZED_TS_MANIFEST."
        )
    manifest = pd.read_csv(manifest)
    manifest = manifest[manifest["dataset"] == "PsiConnect_mean"].copy()
    rows = []
    for row in manifest.itertuples(index=False):
        ts = loadmat(row.timeseries_path)["time_series"][:, valid_mask]
        fc = np.corrcoef(ts, rowvar=False)
        fc = np.clip(fc, -0.999999, 0.999999)
        z = np.arctanh(fc)
        tri = np.triu_indices(z.shape[0], k=1)
        rows.append(
            {
                "SubjectID": row.subject_id,
                "Condition": row.condition,
                "MeanFD": row.mean_fd,
                "FC_global": float(np.nanmean(z[tri])),
            }
        )
    run_df = pd.DataFrame(rows)
    subj = run_df.groupby(["SubjectID", "Condition"], as_index=False).agg(FC_global=("FC_global", "mean"), MeanFD=("MeanFD", "mean"))
    wide = subj.pivot(index="SubjectID", columns="Condition")
    wide.columns = [f"{metric}_{cond}" for metric, cond in wide.columns]
    wide = wide.reset_index()
    wide["Delta_FC_global"] = wide["FC_global_Psilocybin"] - wide["FC_global_Baseline"]
    wide["Delta_MeanFD"] = wide["MeanFD_Psilocybin"] - wide["MeanFD_Baseline"]
    return wide.merge(psiconnect_meq()[["SubjectID", "MEQ30_MEAN", "MEQ30_MYSTICAL", "MEQ30_POSITIVE", "MEQ30_TRANSCEND", "MEQ30_INEFFABILITY"]], on="SubjectID", how="inner")
