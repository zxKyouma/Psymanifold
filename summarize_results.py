from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
CORE = Path(os.environ.get('HARMONIZED_CORE_METRIC_OUTDIR', ROOT / 'outputs' / 'harmonized_psychedelics_core_metrics'))
EXT = Path(os.environ.get('HARMONIZED_EXT_OUTDIR', ROOT / 'outputs' / 'harmonized_psychedelics_extended_metrics'))
OPT = Path(os.environ.get('HARMONIZED_OPTIONAL_OUTDIR', ROOT / 'outputs' / 'harmonized_psychedelics_optional'))
OUT = Path(os.environ.get('HARMONIZED_SUMMARY_OUTDIR', ROOT / 'outputs' / 'harmonized_psychedelics_summary'))
OUT.mkdir(parents=True, exist_ok=True)
PAIR_DEFS = {
    'PsiConnect_mean': [('Baseline', 'Psilocybin', 'acute_psilo')],
    'CHARM_Psilocybin': [('Placebo', 'Psychedelic', 'acute_psilo'), ('Baseline', 'After', 'afterglow')],
    'LSD': [('Placebo', 'Psychedelic', 'acute_lsd')],
    'DMT': [('Placebo', 'Psychedelic', 'acute_dmt')],
}

def write_csv(df, name):
    path = OUT / name
    df.to_csv(path, index=False)
    print('wrote', path, df.shape)

def numeric_metric_columns(df, exclude):
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

def subject_condition_means(df, group_cols, exclude):
    exclude = set(exclude) | set(group_cols)
    metrics = numeric_metric_columns(df, exclude)
    out = df[group_cols + metrics].groupby(group_cols, dropna=False).mean(numeric_only=True).reset_index()
    return out, metrics

def add_fdr(out):
    if out.empty:
        return out
    out['Q_FDR'] = np.nan
    for (dataset, contrast), idx in out.groupby(['Dataset','Contrast']).groups.items():
        p = out.loc[idx, 'P'].to_numpy(dtype=float)
        order = np.argsort(p)
        ranked = p[order]
        m = len(ranked)
        q = np.empty(m, dtype=float)
        prev = 1.0
        for i in range(m - 1, -1, -1):
            val = ranked[i] * m / (i + 1)
            prev = min(prev, val)
            q[i] = prev
        q_full = np.empty(m, dtype=float)
        q_full[order] = np.clip(q, 0, 1)
        out.loc[idx, 'Q_FDR'] = q_full
    return out.sort_values(['Dataset','Contrast','Q_FDR','P','Metric']).reset_index(drop=True)

def paired_contrasts(df, dataset_col, subject_col, condition_col, metrics, extra_keys=None):
    extra_keys = extra_keys or []
    rows = []
    for dataset, defs in PAIR_DEFS.items():
        ds = df[df[dataset_col] == dataset].copy()
        if ds.empty:
            continue
        merge_keys = [subject_col] + extra_keys
        for cond_a, cond_b, label in defs:
            a = ds[ds[condition_col] == cond_a][merge_keys + metrics].copy()
            b = ds[ds[condition_col] == cond_b][merge_keys + metrics].copy()
            if a.empty or b.empty:
                continue
            merged = a.merge(b, on=merge_keys, suffixes=('_a','_b'))
            for metric in metrics:
                xa = pd.to_numeric(merged[f'{metric}_a'], errors='coerce')
                xb = pd.to_numeric(merged[f'{metric}_b'], errors='coerce')
                mask = xa.notna() & xb.notna()
                if mask.sum() < 2:
                    continue
                xa = xa[mask].to_numpy(dtype=float)
                xb = xb[mask].to_numpy(dtype=float)
                t,p = stats.ttest_rel(xb, xa, nan_policy='omit')
                rows.append({'Dataset':dataset,'Contrast':label,'ConditionA':cond_a,'ConditionB':cond_b,'Metric':metric,'N':int(mask.sum()),'MeanA':float(np.nanmean(xa)),'MeanB':float(np.nanmean(xb)),'Delta_B_minus_A':float(np.nanmean(xb-xa)),'T':float(t),'P':float(p)})
    return add_fdr(pd.DataFrame(rows))

def run_core_summary():
    df = pd.read_csv(CORE / 'harmonized_results_summary.csv')
    dyn = pd.read_csv(EXT / 'harmonized_state_dynamics_run_metrics.csv')
    join_keys = ['Dataset', 'SubjectID', 'Session', 'Condition', 'RunLabel', 'FileKey']
    dyn_keep = [c for c in ['SwitchRate', 'Entropy'] if c in dyn.columns]
    if dyn_keep:
        df = df.drop(columns=[c for c in dyn_keep if c in df.columns])
        df = df.merge(dyn[join_keys + dyn_keep], on=join_keys, how='left')
    highres_path = OPT / 'optional_highres_ngsc.csv'
    if highres_path.exists():
        hi = pd.read_csv(highres_path)
        hi_keep = [c for c in ['NGSC_vtx'] if c in hi.columns]
        if hi_keep:
            df = df.drop(columns=[c for c in hi_keep if c in df.columns], errors='ignore')
            df = df.merge(hi[join_keys + hi_keep], on=join_keys, how='left')
    means, metrics = subject_condition_means(
        df,
        ['Dataset','SubjectID','Condition'],
        {'Session','RunLabel','FileKey','TimeSeriesPath','InputPath','NumFrames','NumROIs','MeanFD'},
    )
    write_csv(means, 'harmonized_metric_subject_condition_means.csv')
    write_csv(paired_contrasts(means,'Dataset','SubjectID','Condition',metrics), 'harmonized_metric_paired_contrasts.csv')

def run_state_dynamics_summary():
    df = pd.read_csv(EXT / 'harmonized_state_dynamics_run_metrics.csv')
    means, metrics = subject_condition_means(df, ['Dataset','SubjectID','Condition'], {'Session','RunLabel','FileKey'})
    means = means[['Dataset','SubjectID','Condition'] + metrics]
    write_csv(means, 'harmonized_state_dynamics_subject_condition_means.csv')
    write_csv(paired_contrasts(means,'Dataset','SubjectID','Condition',metrics), 'harmonized_state_dynamics_paired_contrasts.csv')

def run_disp_summary():
    df = pd.read_csv(EXT / 'harmonized_centroid_dispersion_by_run.csv')
    means, metrics = subject_condition_means(df, ['Dataset','SubjectID','Condition'], {'Session','RunLabel','FileKey'})
    write_csv(means, 'harmonized_centroid_dispersion_subject_condition_means.csv')
    write_csv(paired_contrasts(means,'Dataset','SubjectID','Condition',metrics), 'harmonized_centroid_dispersion_paired_contrasts.csv')

def run_energy_summary():
    df = pd.read_csv(EXT / 'harmonized_energy_landscape_metrics.csv')
    means, metrics = subject_condition_means(df, ['Dataset','SubjectID','Condition'], {'Session','RunLabel','FileKey'})
    write_csv(means, 'harmonized_energy_subject_condition_means.csv')
    write_csv(paired_contrasts(means,'Dataset','SubjectID','Condition',metrics), 'harmonized_energy_paired_contrasts.csv')

def run_state_network_summary():
    df = pd.read_csv(EXT / 'harmonized_state_network_correlations.csv')
    means, metrics = subject_condition_means(df, ['Dataset','SubjectID','Condition','StateID','Network'], {'Session','RunLabel','FileKey'})
    write_csv(means, 'harmonized_state_network_subject_condition_means.csv')
    write_csv(paired_contrasts(means,'Dataset','SubjectID','Condition',metrics, extra_keys=['StateID','Network']), 'harmonized_state_network_paired_contrasts.csv')

def run_centroid_pca():
    df = pd.read_csv(EXT / 'harmonized_state_centroids.csv')
    roi_cols = [c for c in df.columns if c.startswith('ROI_')]
    from sklearn.decomposition import PCA
    X = df[roi_cols].to_numpy(dtype=float)
    pca = PCA(n_components=3, random_state=42)
    S = pca.fit_transform(X)
    out = df[['Dataset','SubjectID','Condition','RunLabel','StateID']].copy()
    out['PC1'] = S[:,0]; out['PC2'] = S[:,1]; out['PC3'] = S[:,2]
    write_csv(out, 'harmonized_centroid_pca_scores.csv')

run_core_summary(); run_state_dynamics_summary(); run_disp_summary(); run_energy_summary(); run_state_network_summary(); run_centroid_pca()
