function run_charm_afterglow_network_posthoc()
% Freeze network-level CHARM afterglow post-hoc tests for network-to-global geometry.
%
% Fits, separately within each canonical network:
%   NetToGlobalDist ~ Condition + MeanFDc + (1 | SubjectID)
%
% using the harmonized psychedelic geometry inputs for CHARM_Psilocybin,
% restricted to Baseline and After runs.

ROOT = fileparts(fileparts(mfilename('fullpath')));
indir = fullfile(ROOT, 'outputs', 'harmonized_psychedelics_lme');
outdir = fullfile(ROOT, 'outputs', 'charm_afterglow_omnibus_geometry');
compat_outdir = fullfile(ROOT, 'outputs', 'charm_afterglow_geometry');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
if ~exist(compat_outdir, 'dir')
    mkdir(compat_outdir);
end

global_path = fullfile(indir, 'harmonized_network_to_global_distances_with_fd.csv');
fprintf('Reading %s\n', global_path);
G = readtable(global_path, 'TextType', 'string');

G = G(strcmp(G.Dataset, "CHARM_Psilocybin") & ismember(G.Condition, ["Baseline","After"]), :);
G.MeanFD_final = fallback_meanfd(G);
G = G(:, {'SubjectID','Condition','Network','ProcrustesGlobalDist','MeanFD_final'});
G.Properties.VariableNames{'ProcrustesGlobalDist'} = 'NetToGlobalDist';
G.Properties.VariableNames{'MeanFD_final'} = 'MeanFD';
has_motion = any(~isnan(G.MeanFD));
G = rmmissing(G, 'DataVariables', {'SubjectID','Condition','Network','NetToGlobalDist'});
G = G(strcmp(G.Network, "none") == 0, :);
if has_motion
    G.MeanFDc = G.MeanFD - mean(G.MeanFD, 'omitnan');
else
    G.MeanFDc = zeros(height(G), 1);
end
G.SubjectID = categorical(G.SubjectID);
G.Condition = categorical(G.Condition, {'Baseline','After'});
G.Network = categorical(G.Network);

networks = categories(G.Network);
n = numel(networks);

Network = strings(n,1);
NRows = nan(n,1);
NSubjects = nan(n,1);
BaselineMean = nan(n,1);
AfterMean = nan(n,1);
ConditionEstimate = nan(n,1);
ConditionSE = nan(n,1);
TStat = nan(n,1);
DF = nan(n,1);
PValue = nan(n,1);
MeanFDEstimate = nan(n,1);
MeanFDP = nan(n,1);
if has_motion
    model_formula = "NetToGlobalDist ~ Condition + MeanFDc + (1 | SubjectID)";
else
    model_formula = "NetToGlobalDist ~ Condition + (1 | SubjectID)";
end
Formula = repmat(model_formula, n, 1);

for i = 1:n
    net = networks{i};
    ds = G(G.Network == net, :);
    Network(i) = string(net);
    NRows(i) = height(ds);
    NSubjects(i) = numel(categories(removecats(ds.SubjectID)));

    base = ds.NetToGlobalDist(ds.Condition == 'Baseline');
    aft = ds.NetToGlobalDist(ds.Condition == 'After');
    BaselineMean(i) = mean(base, 'omitnan');
    AfterMean(i) = mean(aft, 'omitnan');

    fprintf('Fitting %s (%d rows, %d subjects)\n', net, height(ds), NSubjects(i));
    lme = fitlme(ds, char(model_formula));
    coef = ensure_table(lme.Coefficients);

    cond_idx = find(strcmp(string(coef.Name), 'Condition_After'));
    fd_idx = find(strcmp(string(coef.Name), 'MeanFDc'));

    if ~isempty(cond_idx)
        ConditionEstimate(i) = coef.Estimate(cond_idx(1));
        ConditionSE(i) = coef.SE(cond_idx(1));
        TStat(i) = coef.tStat(cond_idx(1));
        DF(i) = coef.DF(cond_idx(1));
        PValue(i) = coef.pValue(cond_idx(1));
    end

    if ~isempty(fd_idx)
        MeanFDEstimate(i) = coef.Estimate(fd_idx(1));
        MeanFDP(i) = coef.pValue(fd_idx(1));
    end
end

FDR_Q = bh_fdr(PValue);

out = table(Network, NRows, NSubjects, BaselineMean, AfterMean, ...
    ConditionEstimate, ConditionSE, TStat, DF, PValue, FDR_Q, ...
    MeanFDEstimate, MeanFDP, Formula);
out = sortrows(out, 'PValue');

writetable(out, fullfile(outdir, 'afterglow_network_to_global_posthoc_lme.csv'));
writetable(out, fullfile(compat_outdir, 'charm_afterglow_network_posthoc.csv'));

fprintf('\nSaved %s\n', fullfile(outdir, 'afterglow_network_to_global_posthoc_lme.csv'));
disp(out);
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

function T = ensure_table(x)
if istable(x)
    T = x;
elseif isa(x, 'dataset')
    T = dataset2table(x);
else
    error('Unsupported output type: %s', class(x));
end
end

function q = bh_fdr(p)
p = p(:);
q = nan(size(p));
valid = ~isnan(p);
pv = p(valid);
m = numel(pv);
if m == 0
    return;
end
[ps, order] = sort(pv);
adj = ps .* (m ./ (1:m)');
adj = flipud(cummin(flipud(adj)));
adj(adj > 1) = 1;
q(valid) = adj(invert_permutation(order));
end

function inv = invert_permutation(order)
inv = zeros(size(order));
inv(order) = 1:numel(order);
end
