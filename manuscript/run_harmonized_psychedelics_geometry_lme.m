clear; clc; close all;
fprintf('--- Harmonized Psychedelics Geometry LME ---\n');

ROOT = fileparts(fileparts(mfilename('fullpath')));
CORE_DIR = fullfile(ROOT, 'outputs', 'harmonized_psychedelics_core_metrics');
OUTDIR = fullfile(ROOT, 'outputs', 'harmonized_psychedelics_geometry_lme');
if ~exist(OUTDIR, 'dir'), mkdir(OUTDIR); end

datasets = {
    'PsiConnect_mean', 'acute_psilo', 'Baseline', 'Psilocybin';
    'CHARM_Psilocybin', 'acute_psilo', 'Placebo', 'Psychedelic';
    'CHARM_Psilocybin', 'afterglow', 'Baseline', 'After';
    'LSD', 'acute_lsd', 'Placebo', 'Psychedelic';
};

pair_tbl = readtable(fullfile(CORE_DIR, 'harmonized_network_procrustes_distances.csv'), 'TextType', 'string');
global_tbl = readtable(fullfile(CORE_DIR, 'harmonized_network_to_global_distances.csv'), 'TextType', 'string');

pair_rows = table();
global_rows = table();

for i = 1:size(datasets, 1)
    dataset_name = string(datasets{i, 1});
    analysis_name = string(datasets{i, 2});
    cond_a = string(datasets{i, 3});
    cond_b = string(datasets{i, 4});

    fprintf('Dataset %s | %s | %s -> %s\n', dataset_name, analysis_name, cond_a, cond_b);

    ds_pair = subset_conditions(pair_tbl, dataset_name, cond_a, cond_b);
    pairs = unique(ds_pair.Pair);
    for j = 1:numel(pairs)
        sub = ds_pair(ds_pair.Pair == pairs(j), :);
        row = fit_condition_lme(sub, 'ProcrustesNetDist', dataset_name, analysis_name, cond_a, cond_b, "MeanFD");
        if isempty(row)
            continue;
        end
        row.Family = "pairwise_procrustes";
        row.Pair = pairs(j);
        pair_rows = [pair_rows; row]; %#ok<AGROW>
    end

    ds_global = subset_conditions(global_tbl, dataset_name, cond_a, cond_b);
    nets = unique(ds_global.Network);
    for j = 1:numel(nets)
        sub = ds_global(ds_global.Network == nets(j), :);
        row = fit_condition_lme(sub, 'ProcrustesGlobalDist', dataset_name, analysis_name, cond_a, cond_b, "MeanFD");
        if isempty(row)
            continue;
        end
        row.Family = "global_procrustes";
        row.Network = nets(j);
        global_rows = [global_rows; row]; %#ok<AGROW>
    end
end

pair_rows = add_fdr_by_analysis(pair_rows);
global_rows = add_fdr_by_analysis(global_rows);

write_csv_safe(pair_rows, fullfile(OUTDIR, 'harmonized_pairwise_procrustes_lme_stats.csv'));
write_csv_safe(global_rows, fullfile(OUTDIR, 'harmonized_network_to_global_lme_stats.csv'));

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

    if height(ds) < 4 || numel(unique(ds.Condition)) < 2
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
        fprintf('Geometry LME failed %s %s: %s\n', dataset_name, metric, ME.message);
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
