from pathlib import Path
import os
import numpy as np
import pandas as pd
from analysis_config import DEFAULT_CHARM_MEANFD_CSV

ROOT = Path(__file__).resolve().parent
BASE = Path(os.environ.get('HARMONIZED_BASE_OUTDIR', ROOT / 'outputs' / 'harmonized_psychedelics'))
CORE = Path(os.environ.get('HARMONIZED_CORE_METRIC_OUTDIR', ROOT / 'outputs' / 'harmonized_psychedelics_core_metrics'))
EXT = Path(os.environ.get('HARMONIZED_EXT_OUTDIR', ROOT / 'outputs' / 'harmonized_psychedelics_extended_metrics'))
OPT = Path(os.environ.get('HARMONIZED_OPTIONAL_OUTDIR', ROOT / 'outputs' / 'harmonized_psychedelics_optional'))
OUT = Path(os.environ.get('HARMONIZED_LME_OUTDIR', ROOT / 'outputs' / 'harmonized_psychedelics_lme'))
OUT.mkdir(parents=True, exist_ok=True)

CHARM_MEANFD_CSV = DEFAULT_CHARM_MEANFD_CSV
CHARM_SUB_MAP = {
    'sub-1': 'PS03',
    'sub-2': 'PS02',
    'sub-3': 'PS16',
    'sub-4': 'PS18',
    'sub-5': 'PS19',
    'sub-6': 'PS21',
    'sub-7': 'PS24',
}


def backfill_charm_mean_fd(manifest_df: pd.DataFrame) -> pd.DataFrame:
    out = manifest_df.copy()
    if (
        not str(CHARM_MEANFD_CSV)
        or str(CHARM_MEANFD_CSV) == "."
        or not CHARM_MEANFD_CSV.exists()
        or CHARM_MEANFD_CSV.is_dir()
    ):
        return out

    mean_fd_table = pd.read_csv(CHARM_MEANFD_CSV)
    required_cols = {'SessionID', 'MeanFD'}
    missing_cols = required_cols.difference(mean_fd_table.columns)
    if missing_cols:
        missing = ', '.join(sorted(missing_cols))
        raise ValueError(f'CHARM_MEANFD_CSV is missing required columns: {missing}')
    mean_fd_map = dict(zip(mean_fd_table['SessionID'], mean_fd_table['MeanFD']))

    is_charm = out['dataset'].eq('CHARM_Psilocybin')
    charm_missing = is_charm & out['mean_fd'].isna()
    if not charm_missing.any():
        return out

    session_ids = (
        out.loc[charm_missing, 'subject_id'].map(CHARM_SUB_MAP)
        + '_'
        + out.loc[charm_missing, 'session'].astype(str)
    )
    out.loc[charm_missing, 'mean_fd'] = session_ids.map(mean_fd_map).to_numpy()
    return out


manifest = pd.read_csv(BASE / 'harmonized_manifest.csv')
manifest = backfill_charm_mean_fd(manifest)
if 'mean_fd' in manifest.columns:
    manifest['mean_fd'] = pd.to_numeric(manifest['mean_fd'], errors='coerce')
core = pd.read_csv(CORE / 'harmonized_results_summary.csv')
net = pd.read_csv(CORE / 'harmonized_network_procrustes_distances.csv')
global_net = pd.read_csv(CORE / 'harmonized_network_to_global_distances.csv')
dyn = pd.read_csv(EXT / 'harmonized_state_dynamics_run_metrics.csv')
disp = pd.read_csv(EXT / 'harmonized_centroid_dispersion_by_run.csv')
energy = pd.read_csv(EXT / 'harmonized_energy_landscape_metrics.csv')
state_net = pd.read_csv(EXT / 'harmonized_state_network_correlations.csv')
highres = None
highres_path = OPT / 'optional_highres_ngsc.csv'
if highres_path.exists():
    highres = pd.read_csv(highres_path)

meta = manifest.rename(columns={
    'dataset':'Dataset','subject_id':'SubjectID','session':'Session','condition':'Condition','run_label':'RunLabel','mean_fd':'MeanFD_manifest'
})[['Dataset','SubjectID','Session','Condition','RunLabel','MeanFD_manifest','confounds_path','notes']]
join_keys = ['Dataset','SubjectID','Session','Condition','RunLabel']

def augment(df):
    out = df.merge(meta, on=join_keys, how='left')
    if 'MeanFD' in out.columns:
        out['MeanFD'] = pd.to_numeric(out['MeanFD'], errors='coerce')
    out['MeanFD_manifest'] = pd.to_numeric(out['MeanFD_manifest'], errors='coerce')
    out['MeanFD_final'] = out['MeanFD'] if 'MeanFD' in out.columns else np.nan
    mask = out['MeanFD_final'].isna() & out['MeanFD_manifest'].notna()
    out.loc[mask, 'MeanFD_final'] = out.loc[mask, 'MeanFD_manifest']
    out['MotionWasBackfilled'] = mask.astype(int)
    return out

augment(core).to_csv(OUT / 'harmonized_results_summary_with_fd.csv', index=False)
augment(net).to_csv(OUT / 'harmonized_network_procrustes_distances_with_fd.csv', index=False)
augment(global_net).to_csv(OUT / 'harmonized_network_to_global_distances_with_fd.csv', index=False)
augment(dyn).to_csv(OUT / 'harmonized_state_dynamics_run_metrics_with_fd.csv', index=False)
augment(disp).to_csv(OUT / 'harmonized_centroid_dispersion_by_run_with_fd.csv', index=False)
augment(energy).to_csv(OUT / 'harmonized_energy_landscape_metrics_with_fd.csv', index=False)
augment(state_net).to_csv(OUT / 'harmonized_state_network_correlations_with_fd.csv', index=False)
if highres is not None:
    augment(highres).to_csv(OUT / 'optional_highres_ngsc_with_fd.csv', index=False)
manifest.to_csv(OUT / 'harmonized_manifest_with_fd.csv', index=False)
print('prepared', OUT)
