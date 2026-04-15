# Psychedelic fMRI Pipeline

Multiscale fMRI analysis pipeline accompanying:

> **Multiscale reorganization of functional brain geometry and dynamics under psilocybin** 

This repository contains the harmonized analysis workflow used to characterize
brain dynamics across psilocybin, DMT, and LSD datasets, including manifold
learning (Complex Harmonics Decomposition / CHARM), leading-eigenvector state-dynamics, functional connectivity,
and Procrustes-based network geometry.

---

## Overview

The pipeline operates on Gordon333-parcellated fMRI data and produces:

- Gordon333 mean ROI time series
- Core multiscale metrics: `QED`, `NGSC`, `Q`, `PC`, `FC_within`, `FC_between`,
  network-pair and network-to-global Procrustes distances
- Extended state-dynamics metrics: `SwitchRate`, `Entropy`, occupancy,
  centroids, centroid dispersion, energy summaries
- Mixed-effects model outputs for all core and extension outcomes
- Manuscript result tables (Results 2.1-2.6)

A reduced pathway supports DMT via precomputedROI time series.

---

## Requirements

**Python** (install via pip):
```bash
pip install -r requirements.txt
```

**MATLAB** with the Statistics and Machine Learning Toolbox 
(uses `procrustes`, `kmeans`, `pca`, `fitlme`)

---

## Reproducing Results

The steps below reproduce the multiscale fMRI analyses reported in Results 2.1-2.6 of the manuscript.

> **Note:** Raw neuroimaging data is not bundled. You will need the harmonized
> fMRIPrep derivatives for PsiConnect, the longitudinal Psilocybin cohort, and LSD,
> plus the supported DMT ROI time-series inputs, and the questionnaire files listed under
> [Environment Variables](#environment-variables).

### Step 1 - Build the harmonized dataset manifest
*Aligns participant runs across all four datasets (PsiConnect, Psilocybin, LSD, DMT)
into a single manifest used by all downstream stages.*
```bash
python make_dataset_manifest.py \
  --base-manifest /path/to/base_manifest.csv \
  --lsd-root /path/to/LSD/derivatives/fmriprep_harmonized \
  --dmt-root /path/to/DMT/fMRI
```

### Step 2 - Extract ROI time series
*Extracts Gordon333 parcellated time series, applying FD > 0.3 mm scrubbing
for PsiConnect and LSD by default.*
```bash
python extract_gordon_timeseries.py \
  --manifest outputs/harmonized_psychedelics/harmonized_manifest.csv
```

### Step 3 - Compute core metrics
*Produces QED, NGSC, FC, modularity, and Procrustes geometry: the primary measures underlying Results 2.1, 2.3, and 2.4.*
```matlab
compute_core_metrics
compute_extended_metrics
```

### Step 4 - Compute high-resolution complexity (PsiConnect and Psilocybin cohorts)
*Required for the vertex-level NGSC results reported in Results 2.1.*
```bash
python compute_optional_highres_complexity.py
```

### Step 5 - Prepare mixed-effects inputs
*Merges all metric outputs with manifest metadata and restores CHARM MeanFD
values needed for the afterglow analyses in Results 2.2.*
```bash
python prepare_mixed_effects_inputs.py
```

### Step 6 - Fit mixed-effects models
*Runs the LME models underlying the statistical results across all sections.*
```matlab
fit_core_mixed_effects
fit_extension_mixed_effects
```

### Step 7 - Run secondary analyses and summaries
*Produces subject-delta coupling tables and paired summaries used in Results 2.5
and 2.6.*
```bash
python run_secondary_analyses.py
python summarize_results.py
```

### Step 8 - Generate manuscript result tables
*Writes the final supporting tables for Results 2.1-2.6 into the `outputs/`
directory.*
```bash
python run_manuscript_results.py
```

---

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `GORDON_CIFTI_PATH` | If manifest has dense CIFTI rows | Path to `Gordon333.32k_fs_LR.dlabel.nii` |
| `CHARM_MEANFD_CSV` | For exact afterglow reproduction | Session-level MeanFD (columns: `SessionID`, `MeanFD`) |
| `PSICONNECT_ASC11_TSV` | Manuscript layer only | ASC11 questionnaire data |
| `PSICONNECT_MEQ30_TSV` | Manuscript layer only | MEQ-30 questionnaire data |
| `CHARM_MEQ_CSV` | Manuscript layer only | CHARM cohort MEQ data |

---

## Repository Layout

| File | Description |
|---|---|
| `make_dataset_manifest.py` | Builds harmonized run manifest |
| `extract_gordon_timeseries.py` | Extracts or forwards ROI time series; optional FD scrubbing |
| `compute_core_metrics.m` | Core metric panel (QED, NGSC, FC, Q, PC, Procrustes) |
| `compute_extended_metrics.m` | State dynamics, occupancy, centroids, dispersion |
| `compute_optional_highres_complexity.py` | Vertex-level NGSC for supported datasets |
| `prepare_mixed_effects_inputs.py` | Merges metrics with manifest; restores MeanFD |
| `fit_core_mixed_effects.m` | Core LME models |
| `fit_extension_mixed_effects.m` | Extension LMEs (geometry, peripherality, dispersion) |
| `run_secondary_analyses.py` | Subject-delta coupling and paired summaries |
| `summarize_results.py` | Subject-condition means and summary tables |
| `run_manuscript_results.py` | Optional: regenerates manuscript result tables 2.1?2.6 |
| `analysis_config.py` | Shared Python defaults |
| `audit_release_tree.py` | Pre-commit source-only check |

**Bundled assets:**
- `assets/gordon333MNI.nii.gz`
- `assets/gordon_roi_network_mapping.csv`
- `assets/gordon333CommunityNames.txt`

Raw neuroimaging data is not bundled.

---

## Notes

- State dynamics use leading eigenvectors of sliding-window correlation matrices
  (not classical phase-coherence LEiDA).
- `Entropy` is Shannon entropy of run-level cluster occupancies, not a
  temporal entropy-rate estimate.
- Shared state templates are fit within each dataset on pooled leading-eigenvector
  windows, then all runs are reassigned into that shared space.
- `FC_within`, `FC_between`, `Q`, and `PC` use positive edge weights only;
  Gordon parcels labeled `none` are excluded from network averages.
- Omnibus LMEs use maximum likelihood (`FitMethod = ML`) for fixed-effects
  comparisons.
- Cross-metric coupling is handled as subject-delta correlations, not LMEs.
---
