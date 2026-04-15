function run_charm_afterglow_omnibus_geometry()
% Harmonized CHARM afterglow omnibus geometry models.
%
% Fits:
%   1) ProcrustesGlobalDist ~ Condition * Network + MeanFDc + (1|SubjectID)
%   2) MeanDistToOthers     ~ Condition * Network + MeanFDc + (1|SubjectID)
%
% using the harmonized psychedelic geometry inputs for CHARM_Psilocybin,
% restricted to Baseline and After runs.

ROOT = fileparts(fileparts(mfilename('fullpath')));
indir_lme = fullfile(ROOT, 'outputs', 'harmonized_psychedelics_lme');
outdir = fullfile(ROOT, 'outputs', 'charm_afterglow_omnibus_geometry');
compat_outdir = fullfile(ROOT, 'outputs', 'charm_afterglow_geometry');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
if ~exist(compat_outdir, 'dir')
    mkdir(compat_outdir);
end

global_path = fullfile(indir_lme, 'harmonized_network_to_global_distances_with_fd.csv');
pairwise_path = fullfile(indir_lme, 'harmonized_network_procrustes_distances_with_fd.csv');

fprintf('Reading %s\n', global_path);
G = readtable(global_path, 'TextType', 'string');
fprintf('Reading %s\n', pairwise_path);
P = readtable(pairwise_path, 'TextType', 'string');

%% Network-to-global omnibus model
G = G(strcmp(G.Dataset, "CHARM_Psilocybin") & ismember(G.Condition, ["Baseline","After"]), :);
G.MeanFD_final = fallback_meanfd(G);
G = G(:, {'SubjectID','Condition','Network','ProcrustesGlobalDist','MeanFD_final'});
G.Properties.VariableNames{'ProcrustesGlobalDist'} = 'NetToGlobalDist';
G.Properties.VariableNames{'MeanFD_final'} = 'MeanFD';
has_motion_g = any(~isnan(G.MeanFD));
G = rmmissing(G, 'DataVariables', {'SubjectID','Condition','Network','NetToGlobalDist'});
if has_motion_g
    G.MeanFDc = G.MeanFD - mean(G.MeanFD, 'omitnan');
else
    G.MeanFDc = zeros(height(G), 1);
end
G.SubjectID = categorical(G.SubjectID);
G.Condition = categorical(G.Condition, {'Baseline','After'});
G.Network = categorical(G.Network);

fprintf('Fitting network-to-global omnibus model on %d rows...\n', height(G));
if has_motion_g
    formula_h1 = 'NetToGlobalDist ~ Condition * Network + MeanFDc + (1 | SubjectID)';
else
    formula_h1 = 'NetToGlobalDist ~ Condition * Network + (1 | SubjectID)';
end
lme_h1 = fitlme(G, formula_h1);
anova_h1 = anova(lme_h1);
coef_h1 = lme_h1.Coefficients;

anova_h1 = ensure_table(anova_h1);
coef_h1 = ensure_table(coef_h1);

writetable(anova_h1, fullfile(outdir, 'afterglow_network_to_global_omnibus_anova.csv'));
writetable(coef_h1, fullfile(outdir, 'afterglow_network_to_global_omnibus_coefficients.csv'));

stats_h1 = summarize_terms(anova_h1, 'network_to_global');
writetable(stats_h1, fullfile(outdir, 'afterglow_network_to_global_omnibus_terms.csv'));
writetable(stats_h1, fullfile(compat_outdir, 'charm_afterglow_omnibus_geometry.csv'));

%% Mean distance to others omnibus model
P = P(strcmp(P.Dataset, "CHARM_Psilocybin") & ismember(P.Condition, ["Baseline","After"]), :);
P.MeanFD_final = fallback_meanfd(P);
P = P(:, {'SubjectID','Condition','Pair','ProcrustesNetDist','MeanFD_final'});
P.Properties.VariableNames{'MeanFD_final'} = 'MeanFD';
has_motion_p = any(~isnan(P.MeanFD));
P = rmmissing(P, 'DataVariables', {'SubjectID','Condition','Pair','ProcrustesNetDist'});

parts = split(P.Pair, '-');
P.Network1 = parts(:,1);
P.Network2 = parts(:,2);

H1 = P(:, {'SubjectID','Condition','MeanFD','Network1','ProcrustesNetDist'});
H1.Properties.VariableNames{'Network1'} = 'Network';
H2 = P(:, {'SubjectID','Condition','MeanFD','Network2','ProcrustesNetDist'});
H2.Properties.VariableNames{'Network2'} = 'Network';
H = [H1; H2];

Hub = groupsummary(H, {'SubjectID','Condition','MeanFD','Network'}, 'mean', 'ProcrustesNetDist');
Hub.Properties.VariableNames{'mean_ProcrustesNetDist'} = 'MeanDistToOthers';
Hub = rmmissing(Hub, 'DataVariables', {'SubjectID','Condition','Network','MeanDistToOthers'});
if has_motion_p
    Hub.MeanFDc = Hub.MeanFD - mean(Hub.MeanFD, 'omitnan');
else
    Hub.MeanFDc = zeros(height(Hub), 1);
end
Hub.SubjectID = categorical(Hub.SubjectID);
Hub.Condition = categorical(Hub.Condition, {'Baseline','After'});
Hub.Network = categorical(Hub.Network);

fprintf('Fitting mean-distance-to-others omnibus model on %d rows...\n', height(Hub));
if has_motion_p
    formula_h2 = 'MeanDistToOthers ~ Condition * Network + MeanFDc + (1 | SubjectID)';
else
    formula_h2 = 'MeanDistToOthers ~ Condition * Network + (1 | SubjectID)';
end
lme_h2 = fitlme(Hub, formula_h2);
anova_h2 = anova(lme_h2);
coef_h2 = lme_h2.Coefficients;

anova_h2 = ensure_table(anova_h2);
coef_h2 = ensure_table(coef_h2);

writetable(anova_h2, fullfile(outdir, 'afterglow_mean_dist_to_others_omnibus_anova.csv'));
writetable(coef_h2, fullfile(outdir, 'afterglow_mean_dist_to_others_omnibus_coefficients.csv'));

stats_h2 = summarize_terms(anova_h2, 'mean_dist_to_others');
writetable(stats_h2, fullfile(outdir, 'afterglow_mean_dist_to_others_omnibus_terms.csv'));

%% Metadata
meta = table( ...
    string({'network_to_global'; 'mean_dist_to_others'}), ...
    [height(G); height(Hub)], ...
    [numel(categories(G.SubjectID)); numel(categories(Hub.SubjectID))], ...
    [numel(categories(G.Network)); numel(categories(Hub.Network))], ...
    'VariableNames', {'Model','Rows','Subjects','Networks'});
writetable(meta, fullfile(outdir, 'afterglow_omnibus_model_metadata.csv'));

fprintf('\nSaved omnibus outputs to %s\n', outdir);
disp(stats_h1);
disp(stats_h2);
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

function meanfd = fallback_meanfd(T)
meanfd = nan(height(T), 1);
for name = ["MeanFD_final","MeanFD","MeanFD_manifest"]
    if ismember(name, string(T.Properties.VariableNames))
        candidate = T.(name);
        if iscell(candidate)
            candidate = cellfun(@str2double, candidate);
        end
        if ~isnumeric(candidate)
            candidate = double(candidate);
        end
        replace_idx = isnan(meanfd) & ~isnan(candidate);
        meanfd(replace_idx) = candidate(replace_idx);
    end
end
end

function out = summarize_terms(anova_tbl, model_name)
term_names = string(anova_tbl.Term);
keep = ismember(term_names, ["Condition","Network","Condition:Network","MeanFDc"]);
sub = anova_tbl(keep, :);

out = table();
out.Model = repmat(string(model_name), height(sub), 1);
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
