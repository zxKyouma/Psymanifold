from __future__ import annotations

from pathlib import Path
import os


ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
OUTPUTS_DIR = ROOT / "outputs"
TIMESERIES_DIR = ROOT / "timeseries"

DEFAULT_MANIFEST_DIR = OUTPUTS_DIR / "harmonized_psychedelics"
DEFAULT_CORE_METRIC_DIR = OUTPUTS_DIR / "harmonized_psychedelics_core_metrics"
DEFAULT_EXT_METRIC_DIR = OUTPUTS_DIR / "harmonized_psychedelics_extended_metrics"
DEFAULT_LME_DIR = OUTPUTS_DIR / "harmonized_psychedelics_lme"
DEFAULT_EXT_LME_DIR = OUTPUTS_DIR / "harmonized_psychedelics_extension_lme"
DEFAULT_SUMMARY_DIR = OUTPUTS_DIR / "harmonized_psychedelics_summary"
DEFAULT_PARALLEL_EXT_DIR = OUTPUTS_DIR / "harmonized_psychedelics_parallel_extensions"
DEFAULT_TIMESERIES_DIR = TIMESERIES_DIR / "harmonized_psychedelics"

DEFAULT_GORDON_VOL = ASSETS_DIR / "gordon333MNI.nii.gz"
DEFAULT_GORDON_NETWORK_MAP = ASSETS_DIR / "gordon_roi_network_mapping.csv"
DEFAULT_GORDON_CIFTI = Path(
    os.environ.get(
        "GORDON_CIFTI_PATH",
        "",
    )
)
DEFAULT_CHARM_MEANFD_CSV = Path(
    os.environ.get(
        "CHARM_MEANFD_CSV",
        "",
    )
)
DEFAULT_FD_SCRUB_THRESHOLD = 0.3
DEFAULT_MIN_RETAINED_FRAMES = 120

DATASET_CONTRASTS = {
    "PsiConnect": [("Baseline", "Psilocybin", "acute_psilo")],
    "Psilocybin": [("Placebo", "Psychedelic", "acute_psilo"), ("Baseline", "After", "afterglow")],
    "LSD": [("Placebo", "Psychedelic", "acute_lsd")],
    "DMT": [("Placebo", "Psychedelic", "acute_dmt")],
}

PSILOCYBIN_ALL_CONDITION_ORDER = [
    "Baseline",
    "Between",
    "Placebo",
    "Psychedelic",
    "After",
]

PSILOCYBIN_PAIRWISE_CONTRASTS = [
    ("Placebo", "Psychedelic", "acute_psilo"),
    ("Baseline", "After", "afterglow"),
    ("Baseline", "Between", "between_vs_baseline"),
    ("Baseline", "Psychedelic", "psychedelic_vs_baseline"),
]

CHARM_SUB_MAP = {
    "sub-1": "PS03",
    "sub-2": "PS02",
    "sub-3": "PS16",
    "sub-4": "PS18",
    "sub-5": "PS19",
    "sub-6": "PS21",
    "sub-7": "PS24",
}

FD_SCRUB_DATASETS = {"PsiConnect", "LSD"}
