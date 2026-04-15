clear; clc; close all;
fprintf('--- Harmonized Psychedelics Core LME ---\n');
ROOT = fileparts(mfilename('fullpath'));
INDIR = getenv('HARMONIZED_LME_OUTDIR');
if isempty(INDIR), INDIR = fullfile(ROOT, 'outputs', 'harmonized_psychedelics_lme'); end
OUTDIR = getenv('HARMONIZED_LME_OUTDIR');
if isempty(OUTDIR), OUTDIR = fullfile(ROOT, 'outputs', 'harmonized_psychedelics_lme'); end
T = readtable(fullfile(INDIR, 'harmonized_results_summary_with_fd.csv'), 'TextType', 'string', 'VariableNamingRule', 'preserve');
TDYN = readtable(fullfile(INDIR, 'harmonized_state_dynamics_run_metrics_with_fd.csv'), 'TextType', 'string', 'VariableNamingRule', 'preserve');
metrics = {'QED','NGSC','Q','PC','FC_within','FC_between','SwitchRate','Entropy'};
highres_path = fullfile(INDIR, 'optional_highres_ngsc_with_fd.csv');
THI = table();
if isfile(highres_path)
    THI = readtable(highres_path, 'TextType', 'string', 'VariableNamingRule', 'preserve');
    if ismember('NGSC_vtx', THI.Properties.VariableNames)
        metrics{end+1} = 'NGSC_vtx';
    end
end
rows = table();
for m = 1:numel(metrics)
    metric = metrics{m};
    source_tbl = T;
    if ismember(metric, {'SwitchRate','Entropy'})
        source_tbl = TDYN;
    elseif metric == "NGSC_vtx"
        source_tbl = THI;
    end
    ds = source_tbl(source_tbl.Dataset == "PsiConnect_mean" & ismember(source_tbl.Condition, ["Baseline","Psilocybin"]), :);
    rows = [rows; fit_condition_lme(ds, metric, 'PsiConnect_mean', 'acute_psilo', 'Baseline', 'Psilocybin')]; %#ok<AGROW>
    ds = source_tbl(source_tbl.Dataset == "CHARM_Psilocybin" & ismember(source_tbl.Condition, ["Placebo","Psychedelic"]), :);
    rows = [rows; fit_condition_lme(ds, metric, 'CHARM_Psilocybin', 'acute_psilo', 'Placebo', 'Psychedelic')]; %#ok<AGROW>
    ds = source_tbl(source_tbl.Dataset == "CHARM_Psilocybin" & ismember(source_tbl.Condition, ["Baseline","After"]), :);
    rows = [rows; fit_condition_lme(ds, metric, 'CHARM_Psilocybin', 'afterglow', 'Baseline', 'After')]; %#ok<AGROW>
    ds = source_tbl(source_tbl.Dataset == "LSD" & ismember(source_tbl.Condition, ["Placebo","Psychedelic"]), :);
    rows = [rows; fit_condition_lme(ds, metric, 'LSD', 'acute_lsd', 'Placebo', 'Psychedelic')]; %#ok<AGROW>
    ds = source_tbl(source_tbl.Dataset == "DMT" & ismember(source_tbl.Condition, ["Placebo","Psychedelic"]), :);
    rows = [rows; fit_condition_lme(ds, metric, 'DMT', 'acute_dmt', 'Placebo', 'Psychedelic')]; %#ok<AGROW>
end
writetable(rows, fullfile(OUTDIR, 'harmonized_core_lme_stats.csv'));
fprintf('Wrote %s with %d rows\n', fullfile(OUTDIR, 'harmonized_core_lme_stats.csv'), height(rows));

function out = fit_condition_lme(ds, metric, dataset_name, analysis_name, cond_a, cond_b)
    out = table(); if isempty(ds), return; end
    ds = ds(isfinite(ds.(metric)), :); if height(ds) < 4, return; end
    ds.SubjectID = categorical(ds.SubjectID);
    ds.Condition = categorical(string(ds.Condition), [string(cond_a), string(cond_b)]);
    use_motion = sum(isfinite(ds.MeanFD_final)) >= 4;
    if use_motion
        ds = ds(isfinite(ds.MeanFD_final), :);
        ds.MeanFDc = ds.MeanFD_final - mean(ds.MeanFD_final, 'omitnan');
        formula = sprintf('%s ~ Condition + MeanFDc + (1|SubjectID)', metric);
    else
        formula = sprintf('%s ~ Condition + (1|SubjectID)', metric);
    end
    try
        lme = fitlme(ds, formula);
        ct = lme.Coefficients; target = "Condition_" + cond_b;
        idx = find(string(ct.Name) == target, 1);
        if isempty(idx), idx = find(contains(string(ct.Name), "Condition") & contains(string(ct.Name), cond_b), 1); end
        if isempty(idx), return; end
        out = table();
        out.Dataset = string(dataset_name); out.Analysis = string(analysis_name); out.Metric = string(metric); out.Model = string(formula); out.Term = string(ct.Name(idx));
        out.NRows = height(ds); out.NSubjects = numel(unique(ds.SubjectID)); out.RefCondition = string(cond_a); out.EffectCondition = string(cond_b);
        out.Estimate = ct.Estimate(idx); out.SE = ct.SE(idx); out.DF = ct.DF(idx); out.TStat = ct.tStat(idx); out.PValue = ct.pValue(idx);
        out.MeanA = mean(ds.(metric)(ds.Condition == cond_a), 'omitnan'); out.MeanB = mean(ds.(metric)(ds.Condition == cond_b), 'omitnan');
    catch ME
        fprintf('LME failed %s %s: %s\n', dataset_name, metric, ME.message);
    end
end
