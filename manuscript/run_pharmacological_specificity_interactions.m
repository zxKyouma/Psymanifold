function run_pharmacological_specificity_interactions()
% Direct cross-dataset interaction tests for macro-scale acute psychedelic metrics.
%
% Fits harmonized acute models on:
%   PsiConnect_mean: Baseline vs Psilocybin
%   CHARM_Psilocybin: Placebo vs Psychedelic
%   LSD: Placebo vs Psychedelic
%
% Main omnibus model:
%   Metric ~ DrugClass * ConditionH + MeanFDc + (1|SubjectKey)
%
% Pairwise follow-up models:
%   Metric ~ Dataset * ConditionH + MeanFDc + (1|SubjectKey)
%   for:
%     1) LSD vs PsiConnect_mean
%     2) LSD vs CHARM_Psilocybin
%     3) PsiConnect_mean vs CHARM_Psilocybin

ROOT = fileparts(fileparts(mfilename('fullpath')));
infile = fullfile(ROOT, 'outputs', 'harmonized_psychedelics_core_metrics', 'harmonized_results_summary.csv');
outdir = fullfile(ROOT, 'outputs', 'pharmacological_specificity_interactions');
compat_outdir = fullfile(ROOT, 'outputs', 'pharmacological_specificity');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
if ~exist(compat_outdir, 'dir')
    mkdir(compat_outdir);
end

T = readtable(infile, 'TextType', 'string', 'VariableNamingRule', 'preserve');
T = subset_acute_rows(T);
T.SubjectKey = categorical(T.Dataset + "__" + T.SubjectID);

metrics = {'QED','NGSC'};
all_drugclass_terms = table();
all_pairwise_terms = table();
all_drugclass_coefs = table();
all_pairwise_coefs = table();
all_means = table();

for m = 1:numel(metrics)
    metric = metrics{m};
    D = T(:, {'Dataset','SubjectID','SubjectKey','ConditionH','MeanFD', metric});
    D = rmmissing(D, 'DataVariables', {metric, 'Dataset', 'ConditionH', 'SubjectKey'});
    D.Dataset = categorical(string(D.Dataset), ["LSD","PsiConnect_mean","CHARM_Psilocybin"]);
    D.ConditionH = categorical(string(D.ConditionH), ["Control","Acute"]);
    D.MeanFDc = D.MeanFD - mean(D.MeanFD, 'omitnan');

    means = groupsummary(D, {'Dataset','ConditionH'}, 'mean', metric);
    means.Properties.VariableNames{end} = 'MeanMetric';
    means.Metric = repmat(string(metric), height(means), 1);
    all_means = [all_means; means(:, {'Metric','Dataset','ConditionH','GroupCount','MeanMetric'})]; %#ok<AGROW>

    % Drug-class omnibus interaction: pooled psilocybin datasets vs LSD.
    G = D;
    G.DrugClass = strings(height(G), 1);
    G.DrugClass(ismember(string(G.Dataset), ["PsiConnect_mean","CHARM_Psilocybin"])) = "Psilocybin";
    G.DrugClass(string(G.Dataset) == "LSD") = "LSD";
    G.DrugClass = categorical(string(G.DrugClass), ["LSD","Psilocybin"]);
    formula_drug = sprintf('%s ~ DrugClass * ConditionH + MeanFDc + (1|SubjectKey)', metric);
    lme_g = fitlme(G, formula_drug);
    a_g = ensure_table(anova(lme_g));
    c_g = ensure_table(lme_g.Coefficients);

    drug_terms = summarize_interaction_terms(a_g, metric, "drugclass_psilo_vs_lsd", ["DrugClass","ConditionH","DrugClass:ConditionH","ConditionH:DrugClass","MeanFDc"]);
    all_drugclass_terms = [all_drugclass_terms; drug_terms]; %#ok<AGROW>

    c_g.Metric = repmat(string(metric), height(c_g), 1);
    c_g.ModelLabel = repmat("drugclass_psilo_vs_lsd", height(c_g), 1);
    all_drugclass_coefs = [all_drugclass_coefs; c_g]; %#ok<AGROW>

    % Pairwise interaction follow-ups.
    pair_defs = {
        "LSD_vs_PsiConnect", ["LSD","PsiConnect_mean"];
        "LSD_vs_CHARM",      ["LSD","CHARM_Psilocybin"];
        };

    for i = 1:size(pair_defs, 1)
        label = pair_defs{i, 1};
        pair_ds = pair_defs{i, 2};
        P = D(ismember(string(D.Dataset), pair_ds), :);
        if numel(unique(P.Dataset)) < 2
            continue;
        end
        P.Dataset = removecats(categorical(string(P.Dataset), pair_ds));
        P.ConditionH = categorical(string(P.ConditionH), ["Control","Acute"]);
        P.SubjectKey = categorical(string(P.SubjectKey));
        P.MeanFDc = P.MeanFD - mean(P.MeanFD, 'omitnan');

        formula_pair = sprintf('%s ~ Dataset * ConditionH + MeanFDc + (1|SubjectKey)', metric);
        try
            lme_p = fitlme(P, formula_pair);
            a_p = ensure_table(anova(lme_p));
            c_p = ensure_table(lme_p.Coefficients);

            pair_terms = summarize_interaction_terms(a_p, metric, label, ["Dataset","ConditionH","Dataset:ConditionH","ConditionH:Dataset","MeanFDc"]);
            all_pairwise_terms = [all_pairwise_terms; pair_terms]; %#ok<AGROW>

            c_p.Metric = repmat(string(metric), height(c_p), 1);
            c_p.ModelLabel = repmat(label, height(c_p), 1);
            all_pairwise_coefs = [all_pairwise_coefs; c_p]; %#ok<AGROW>
        catch ME
            fprintf('Pairwise model failed for %s %s: %s\n', label, metric, ME.message);
        end
    end
end

writetable(all_means, fullfile(outdir, 'metric_means_by_dataset_condition.csv'));
writetable(all_drugclass_terms, fullfile(outdir, 'drugclass_interaction_terms.csv'));
writetable(all_pairwise_terms, fullfile(outdir, 'pairwise_interaction_terms.csv'));
writetable(all_drugclass_coefs, fullfile(outdir, 'drugclass_coefficients.csv'));
writetable(all_pairwise_coefs, fullfile(outdir, 'pairwise_coefficients.csv'));

compat = [all_drugclass_terms; all_pairwise_terms]; %#ok<AGROW>
writetable(compat, fullfile(compat_outdir, 'pharmacological_specificity_interactions.csv'));

fprintf('Saved outputs to %s\n', outdir);
disp(all_drugclass_terms);
disp(all_pairwise_terms);
end

function T = subset_acute_rows(T)
keep = ...
    (T.Dataset == "PsiConnect_mean" & ismember(T.Condition, ["Baseline","Psilocybin"])) | ...
    (T.Dataset == "CHARM_Psilocybin" & ismember(T.Condition, ["Placebo","Psychedelic"])) | ...
    (T.Dataset == "LSD" & ismember(T.Condition, ["Placebo","Psychedelic"]));
T = T(keep, :);
T.ConditionH = strings(height(T), 1);
is_control = ...
    (T.Dataset == "PsiConnect_mean" & T.Condition == "Baseline") | ...
    ((T.Dataset == "CHARM_Psilocybin" | T.Dataset == "LSD") & T.Condition == "Placebo");
is_acute = ...
    (T.Dataset == "PsiConnect_mean" & T.Condition == "Psilocybin") | ...
    ((T.Dataset == "CHARM_Psilocybin" | T.Dataset == "LSD") & T.Condition == "Psychedelic");
T.ConditionH(is_control) = "Control";
T.ConditionH(is_acute) = "Acute";
T = T(T.ConditionH ~= "", :);
end

function T = ensure_table(x)
if istable(x)
    T = x;
elseif isa(x, 'dataset')
    T = dataset2table(x);
else
    error('Unsupported output type: %s', class(x));
end
end

function out = summarize_interaction_terms(anova_tbl, metric, model_label, keep_terms)
term_names = string(anova_tbl.Term);
keep = ismember(term_names, keep_terms);
sub = anova_tbl(keep, :);

out = table();
out.Metric = repmat(string(metric), height(sub), 1);
out.ModelLabel = repmat(string(model_label), height(sub), 1);
out.Term = string(sub.Term);

if ismember('DF1', sub.Properties.VariableNames)
    out.DF1 = sub.DF1;
else
    out.DF1 = nan(height(sub), 1);
end
if ismember('DF2', sub.Properties.VariableNames)
    out.DF2 = sub.DF2;
else
    out.DF2 = nan(height(sub), 1);
end

if ismember('FStat', sub.Properties.VariableNames)
    out.FStat = sub.FStat;
elseif ismember('Stat', sub.Properties.VariableNames)
    out.FStat = sub.Stat;
else
    out.FStat = nan(height(sub), 1);
end

if ismember('pValue', sub.Properties.VariableNames)
    out.pValue = sub.pValue;
else
    out.pValue = nan(height(sub), 1);
end
end
