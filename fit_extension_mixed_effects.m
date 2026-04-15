clear; clc; close all;
fprintf('--- Harmonized Psychedelics Extension LME ---\n');

ROOT = fileparts(mfilename('fullpath'));
LME_DIR = getenv('HARMONIZED_LME_OUTDIR');
if isempty(LME_DIR), LME_DIR = fullfile(ROOT, 'outputs', 'harmonized_psychedelics_lme'); end
PARX_DIR = getenv('HARMONIZED_PARALLEL_EXT_OUTDIR');
if isempty(PARX_DIR), PARX_DIR = fullfile(ROOT, 'outputs', 'harmonized_psychedelics_parallel_extensions'); end
OUTDIR = getenv('HARMONIZED_EXT_LME_OUTDIR');
if isempty(OUTDIR), OUTDIR = fullfile(ROOT, 'outputs', 'harmonized_psychedelics_extension_lme'); end
if ~exist(OUTDIR, 'dir'), mkdir(OUTDIR); end

datasets = {
    'PsiConnect', 'acute_psilo', 'Baseline', 'Psilocybin';
    'Psilocybin', 'acute_psilo', 'Placebo', 'Psychedelic';
    'Psilocybin', 'afterglow', 'Baseline', 'After';
    'LSD', 'acute_lsd', 'Placebo', 'Psychedelic';
    'DMT', 'acute_dmt', 'Placebo', 'Psychedelic';
};

disp_tbl = readtable(fullfile(LME_DIR, 'harmonized_centroid_dispersion_by_run_with_fd.csv'), 'TextType', 'string');
energy_tbl = readtable(fullfile(LME_DIR, 'harmonized_energy_landscape_metrics_with_fd.csv'), 'TextType', 'string');
state_tbl = readtable(fullfile(LME_DIR, 'harmonized_state_network_correlations_with_fd.csv'), 'TextType', 'string');
per_tbl = readtable(fullfile(PARX_DIR, 'harmonized_network_peripherality_by_run.csv'), 'TextType', 'string');
core_tbl = readtable(fullfile(LME_DIR, 'harmonized_results_summary_with_fd.csv'), 'TextType', 'string');
netpair_tbl = readtable(fullfile(LME_DIR, 'harmonized_network_procrustes_distances_with_fd.csv'), 'TextType', 'string');
netglobal_tbl = readtable(fullfile(LME_DIR, 'harmonized_network_to_global_distances_with_fd.csv'), 'TextType', 'string');

% Backfill MeanFD into peripherality rows using the run-level core table.
per_tbl = outerjoin( ...
    per_tbl, ...
    core_tbl(:, {'Dataset','SubjectID','Condition','RunLabel','FileKey','MeanFD_final'}), ...
    'Keys', {'Dataset','SubjectID','Condition','RunLabel','FileKey'}, ...
    'MergeKeys', true, ...
    'Type', 'left');

disp_metrics = {'RawDispersion','PCADispersion','PC1_VarExplained','PC2_VarExplained','PC3_VarExplained'};
energy_metrics = {'EnergyMean','EnergySD','StationaryMassDeepWells','StationaryMassHighPeaks','WellSpeed','HillSpeed','RadiusGyration','QED'};

disp_rows = table();
energy_rows = table();
per_rows = table();
per_omnibus = table();
pair_rows = table();
pair_omnibus = table();
global_rows = table();
global_omnibus = table();
state_rows = table();
state_omnibus = table();

for i = 1:size(datasets, 1)
    dataset_name = string(datasets{i, 1});
    analysis_name = string(datasets{i, 2});
    cond_a = string(datasets{i, 3});
    cond_b = string(datasets{i, 4});

    fprintf('Dataset %s | %s | %s -> %s\n', dataset_name, analysis_name, cond_a, cond_b);

    ds_disp = subset_conditions(disp_tbl, dataset_name, cond_a, cond_b);
    for m = 1:numel(disp_metrics)
        disp_rows = [disp_rows; fit_condition_lme(ds_disp, disp_metrics{m}, dataset_name, analysis_name, cond_a, cond_b, "MeanFD_final")]; %#ok<AGROW>
    end

    ds_energy = subset_conditions(energy_tbl, dataset_name, cond_a, cond_b);
    for m = 1:numel(energy_metrics)
        energy_rows = [energy_rows; fit_condition_lme(ds_energy, energy_metrics{m}, dataset_name, analysis_name, cond_a, cond_b, "MeanFD_final")]; %#ok<AGROW>
    end

    ds_per = subset_conditions(per_tbl, dataset_name, cond_a, cond_b);
    [omni_row, per_net_rows] = fit_network_interaction_lme(ds_per, 'Peripherality', 'Network', dataset_name, analysis_name, cond_a, cond_b, "MeanFD_final");
    per_omnibus = [per_omnibus; omni_row]; %#ok<AGROW>
    per_rows = [per_rows; per_net_rows]; %#ok<AGROW>

    ds_pairs = subset_conditions(netpair_tbl, dataset_name, cond_a, cond_b);
    [pair_omni_row, pair_simple_rows] = fit_network_interaction_lme(ds_pairs, 'ProcrustesNetDist', 'Pair', dataset_name, analysis_name, cond_a, cond_b, "MeanFD_final");
    pair_omnibus = [pair_omnibus; pair_omni_row]; %#ok<AGROW>
    pair_rows = [pair_rows; pair_simple_rows]; %#ok<AGROW>

    ds_global = subset_conditions(netglobal_tbl, dataset_name, cond_a, cond_b);
    [global_omni_row, global_simple_rows] = fit_network_interaction_lme(ds_global, 'ProcrustesGlobalDist', 'Network', dataset_name, analysis_name, cond_a, cond_b, "MeanFD_final");
    global_omnibus = [global_omnibus; global_omni_row]; %#ok<AGROW>
    global_rows = [global_rows; global_simple_rows]; %#ok<AGROW>

    ds_state = subset_conditions(state_tbl, dataset_name, cond_a, cond_b);
    [state_omni_row, state_net_rows] = fit_state_network_lme(ds_state, dataset_name, analysis_name, cond_a, cond_b);
    state_omnibus = [state_omnibus; state_omni_row]; %#ok<AGROW>
    state_rows = [state_rows; state_net_rows]; %#ok<AGROW>
end

disp_rows = add_fdr_by_analysis(disp_rows);
energy_rows = add_fdr_by_analysis(energy_rows);
per_rows = add_fdr_by_analysis(per_rows);
pair_rows = add_fdr_by_analysis(pair_rows);
global_rows = add_fdr_by_analysis(global_rows);
state_rows = add_fdr_by_analysis(state_rows);
per_omnibus = add_fdr_dataset(per_omnibus);
pair_omnibus = add_fdr_dataset(pair_omnibus);
global_omnibus = add_fdr_dataset(global_omnibus);
state_omnibus = add_fdr_dataset(state_omnibus);

write_csv_safe(disp_rows, fullfile(OUTDIR, 'harmonized_dispersion_lme_stats.csv'));
write_csv_safe(energy_rows, fullfile(OUTDIR, 'harmonized_energy_lme_stats.csv'));
write_csv_safe(per_rows, fullfile(OUTDIR, 'harmonized_peripherality_lme_stats.csv'));
write_csv_safe(pair_rows, fullfile(OUTDIR, 'harmonized_network_pair_lme_stats.csv'));
write_csv_safe(global_rows, fullfile(OUTDIR, 'harmonized_network_to_global_lme_stats.csv'));
write_csv_safe(state_rows, fullfile(OUTDIR, 'harmonized_state_network_lme_stats.csv'));
write_csv_safe(per_omnibus, fullfile(OUTDIR, 'harmonized_peripherality_omnibus_lme.csv'));
write_csv_safe(pair_omnibus, fullfile(OUTDIR, 'harmonized_network_pair_omnibus_lme.csv'));
write_csv_safe(global_omnibus, fullfile(OUTDIR, 'harmonized_network_to_global_omnibus_lme.csv'));
write_csv_safe(state_omnibus, fullfile(OUTDIR, 'harmonized_state_network_omnibus_lme.csv'));

fprintf('Done. Outputs written to %s\n', OUTDIR);

function ds = subset_conditions(T, dataset_name, cond_a, cond_b)
    ds = T(T.Dataset == dataset_name & ismember(string(T.Condition), [cond_a, cond_b]), :);
end

function out = fit_condition_lme(ds, metric, dataset_name, analysis_name, cond_a, cond_b, meanfd_var)
    out = table();
    if isempty(ds) || ~ismember(metric, ds.Properties.VariableNames)
        return;
    end
    ds = ds(isfinite(ds.(metric)), :);
    if height(ds) < 4
        return;
    end
    ds.SubjectID = categorical(ds.SubjectID);
    ds.Condition = categorical(string(ds.Condition), [string(cond_a), string(cond_b)]);

    use_motion = false;
    if ismember(meanfd_var, ds.Properties.VariableNames)
        use_motion = sum(isfinite(ds.(meanfd_var))) >= 4;
        if use_motion
            ds = ds(isfinite(ds.(meanfd_var)), :);
            ds.MeanFDc = ds.(meanfd_var) - mean(ds.(meanfd_var), 'omitnan');
        end
    end

    if height(ds) < 4
        return;
    end

    if use_motion
        formula = sprintf('%s ~ Condition + MeanFDc + (1|SubjectID)', metric);
    else
        formula = sprintf('%s ~ Condition + (1|SubjectID)', metric);
    end

    try
        lme = fitlme(ds, formula);
        ct = lme.Coefficients;
        target = "Condition_" + cond_b;
        idx = find(string(ct.Name) == target, 1);
        if isempty(idx)
            idx = find(contains(string(ct.Name), "Condition") & contains(string(ct.Name), cond_b), 1);
        end
        if isempty(idx)
            return;
        end
        out = table();
        out.Dataset = string(dataset_name);
        out.Analysis = string(analysis_name);
        out.Family = "metric";
        out.Metric = string(metric);
        out.Model = string(formula);
        out.Term = string(ct.Name(idx));
        out.NRows = height(ds);
        out.NSubjects = numel(unique(ds.SubjectID));
        out.RefCondition = string(cond_a);
        out.EffectCondition = string(cond_b);
        out.MeanA = mean(ds.(metric)(ds.Condition == cond_a), 'omitnan');
        out.MeanB = mean(ds.(metric)(ds.Condition == cond_b), 'omitnan');
        out.Estimate = ct.Estimate(idx);
        out.SE = ct.SE(idx);
        out.DF = ct.DF(idx);
        out.TStat = ct.tStat(idx);
        out.PValue = ct.pValue(idx);
        out.UsedMotion = double(use_motion);
    catch ME
        fprintf('Metric LME failed %s %s: %s\n', dataset_name, metric, ME.message);
    end
end

function [omni_row, rows] = fit_network_interaction_lme(ds, value_var, network_var, dataset_name, analysis_name, cond_a, cond_b, meanfd_var)
    omni_row = table();
    rows = table();
    if isempty(ds) || height(ds) < 8
        return;
    end
    ds = ds(isfinite(ds.(value_var)), :);
    ds.SubjectID = categorical(ds.SubjectID);
    ds.Condition = categorical(string(ds.Condition), [string(cond_a), string(cond_b)]);
    ds.(network_var) = categorical(string(ds.(network_var)));

    use_motion = ismember(meanfd_var, ds.Properties.VariableNames) && sum(isfinite(ds.(meanfd_var))) >= 4;
    if use_motion
        ds = ds(isfinite(ds.(meanfd_var)), :);
        ds.MeanFDc = ds.(meanfd_var) - mean(ds.(meanfd_var), 'omitnan');
        formula_red = sprintf('%s ~ Condition + %s + MeanFDc + (1|SubjectID)', value_var, network_var);
        formula_full = sprintf('%s ~ Condition * %s + MeanFDc + (1|SubjectID)', value_var, network_var);
    else
        formula_red = sprintf('%s ~ Condition + %s + (1|SubjectID)', value_var, network_var);
        formula_full = sprintf('%s ~ Condition * %s + (1|SubjectID)', value_var, network_var);
    end

    try
        mdl_red = fitlme(ds, formula_red, 'FitMethod', 'ML');
        mdl_full = fitlme(ds, formula_full, 'FitMethod', 'ML');
        cmp = compare(mdl_red, mdl_full);
        omni_row = table();
        omni_row.Dataset = string(dataset_name);
        omni_row.Analysis = string(analysis_name);
        omni_row.Family = string(network_var) + "_interaction";
        omni_row.Outcome = string(value_var);
        omni_row.Term = "Condition x " + string(network_var);
        omni_row.ModelReduced = string(formula_red);
        omni_row.ModelFull = string(formula_full);
        omni_row.NRows = height(ds);
        omni_row.NSubjects = numel(unique(ds.SubjectID));
        omni_row.RefCondition = string(cond_a);
        omni_row.EffectCondition = string(cond_b);
        omni_row.LRStat = cmp.LRStat(2);
        omni_row.DF = cmp.deltaDF(2);
        omni_row.PValue = cmp.pValue(2);
        omni_row.UsedMotion = double(use_motion);
    catch ME
        fprintf('Network omnibus LME failed %s: %s\n', dataset_name, ME.message);
    end

    nets = categories(ds.(network_var));
    for i = 1:numel(nets)
        net = string(nets{i});
        sub = ds(ds.(network_var) == net, :);
        row = fit_condition_lme(sub, value_var, dataset_name, analysis_name, cond_a, cond_b, meanfd_var);
        if isempty(row)
            continue;
        end
        row.Family = string(network_var) + "_simple";
        row.(char(network_var)) = net;
        rows = [rows; row]; %#ok<AGROW>
    end
end

function [omni_row, rows] = fit_state_network_lme(ds, dataset_name, analysis_name, cond_a, cond_b)
    omni_row = table();
    rows = table();
    value_var = 'Correlation';
    if isempty(ds) || height(ds) < 20
        return;
    end
    ds = ds(isfinite(ds.(value_var)), :);
    ds.SubjectID = categorical(ds.SubjectID);
    ds.Condition = categorical(string(ds.Condition), [string(cond_a), string(cond_b)]);
    ds.StateID = categorical(ds.StateID);
    ds.Network = categorical(string(ds.Network));

    meanfd_var = "MeanFD_final";
    if ~ismember(meanfd_var, ds.Properties.VariableNames)
        meanfd_var = "MeanFD";
    end
    use_motion = ismember(meanfd_var, ds.Properties.VariableNames) && sum(isfinite(ds.(meanfd_var))) >= 4;
    if use_motion
        ds = ds(isfinite(ds.(meanfd_var)), :);
        ds.MeanFDc = ds.(meanfd_var) - mean(ds.(meanfd_var), 'omitnan');
        formula_red = 'Correlation ~ Condition + StateID + Network + Condition:StateID + Condition:Network + StateID:Network + MeanFDc + (1|SubjectID)';
        formula_full = 'Correlation ~ Condition * StateID * Network + MeanFDc + (1|SubjectID)';
    else
        formula_red = 'Correlation ~ Condition + StateID + Network + Condition:StateID + Condition:Network + StateID:Network + (1|SubjectID)';
        formula_full = 'Correlation ~ Condition * StateID * Network + (1|SubjectID)';
    end

    try
        mdl_red = fitlme(ds, formula_red, 'FitMethod', 'ML');
        mdl_full = fitlme(ds, formula_full, 'FitMethod', 'ML');
        cmp = compare(mdl_red, mdl_full);
        omni_row = table();
        omni_row.Dataset = string(dataset_name);
        omni_row.Analysis = string(analysis_name);
        omni_row.Family = "state_network_interaction";
        omni_row.Outcome = "Correlation";
        omni_row.Term = "Condition x StateID x Network";
        omni_row.ModelReduced = string(formula_red);
        omni_row.ModelFull = string(formula_full);
        omni_row.NRows = height(ds);
        omni_row.NSubjects = numel(unique(ds.SubjectID));
        omni_row.RefCondition = string(cond_a);
        omni_row.EffectCondition = string(cond_b);
        omni_row.LRStat = cmp.LRStat(2);
        omni_row.DF = cmp.deltaDF(2);
        omni_row.PValue = cmp.pValue(2);
        omni_row.UsedMotion = double(use_motion);
    catch ME
        fprintf('State-network omnibus LME failed %s: %s\n', dataset_name, ME.message);
    end

    states = categories(ds.StateID);
    nets = categories(ds.Network);
    for i = 1:numel(states)
        for j = 1:numel(nets)
            sub = ds(ds.StateID == states{i} & ds.Network == nets{j}, :);
            row = fit_condition_lme(sub, value_var, dataset_name, analysis_name, cond_a, cond_b, meanfd_var);
            if isempty(row)
                continue;
            end
            row.Family = "state_network_simple";
            row.StateID = string(states{i});
            row.Network = string(nets{j});
            rows = [rows; row]; %#ok<AGROW>
        end
    end
end

function out = add_fdr_by_analysis(T)
    out = T;
    if isempty(out) || ~ismember('PValue', out.Properties.VariableNames)
        return;
    end
    out.FDR_Q = nan(height(out), 1);
    G = findgroups(out(:, {'Dataset','Analysis','Family'}));
    for g = unique(G(:))'
        idx = find(G == g);
        out.FDR_Q(idx) = bh_adjust(out.PValue(idx));
    end
end

function out = add_fdr_dataset(T)
    out = T;
    if isempty(out) || ~ismember('PValue', out.Properties.VariableNames)
        return;
    end
    out.FDR_Q = nan(height(out), 1);
    G = findgroups(out(:, {'Dataset','Family'}));
    for g = unique(G(:))'
        idx = find(G == g);
        out.FDR_Q(idx) = bh_adjust(out.PValue(idx));
    end
end

function q = bh_adjust(p)
    p = double(p(:));
    q = nan(size(p));
    ok = isfinite(p);
    pv = p(ok);
    if isempty(pv)
        return;
    end
    [ranked, order] = sort(pv);
    m = numel(ranked);
    adj = ranked .* m ./ (1:m)';
    adj = flipud(cummin(flipud(adj)));
    adj = min(adj, 1);
    qv = nan(size(ranked));
    qv(order) = adj;
    q(ok) = qv;
end

function write_csv_safe(tbl, filepath)
    T = tbl;
    for i = 1:width(T)
        v = T.Properties.VariableNames{i};
        if isstring(T.(v))
            T.(v) = cellstr(T.(v));
        elseif iscategorical(T.(v))
            T.(v) = cellstr(string(T.(v)));
        end
    end
    writetable(T, filepath);
    fprintf('  -> Wrote %s (%d rows, %d cols)\n', filepath, height(T), width(T));
end
