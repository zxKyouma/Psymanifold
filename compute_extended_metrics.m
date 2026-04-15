% =========================================================================
% HARMONIZED EXTENDED METRIC RUNNER
%
% Computes the remaining common downstream analyses on the harmonized
% mean ROI time-series manifest:
%   - shared-state leading-eigenvector occupancy / centroids
%   - shared-state-to-network correlations
%   - centroid dispersion
%   - QDM energy-landscape summaries
%
% Important note:
% Shared state templates are fit separately within each dataset using pooled
% leading eigenvectors from all runs, with a per-run sampling cap so no
% single run dominates the shared state space. State-specific inference is
% based only on these shared templates.
% =========================================================================

clear; clc; close all; rng(42);

fprintf('--- Harmonized Extended Metric Runner ---\n');

ROOT = fileparts(mfilename('fullpath'));
MANIFEST_PATH = getenv_default('HARMONIZED_TS_MANIFEST', fullfile(ROOT, 'outputs', 'harmonized_psychedelics', 'harmonized_timeseries_manifest.tsv'));
ROI_NETWORK_MAP_PATH = getenv_default('ROI_NETWORK_MAP_PATH', fullfile(ROOT, 'assets', 'gordon_roi_network_mapping.csv'));
OUTDIR = getenv_default('HARMONIZED_EXT_OUTDIR', fullfile(ROOT, 'outputs', 'harmonized_psychedelics_extended_metrics'));
MAX_ROWS = str2double(getenv_default('HARMONIZED_MAX_ROWS', '0'));
ensure_dir(OUTDIR);

QDM_PARAMS.LATDIM = 18;
QDM_PARAMS.Thorizont_quantum = 2;
QDM_PARAMS.epsilon_quantum = 70;
STATE_WLEN = 30;
STATE_STEP = 1;
STATE_K = 6;
STATE_MAX_WINDOWS_PER_RUN = 200;

fprintf('Loading harmonized manifest...\n');
manifest = readtable(MANIFEST_PATH, 'FileType', 'text', 'Delimiter', '\t', 'TextType', 'string', 'VariableNamingRule', 'preserve');
fprintf('  Rows before limit: %d\n', height(manifest));
if isfinite(MAX_ROWS) && MAX_ROWS > 0
    manifest = manifest(1:min(height(manifest), MAX_ROWS), :);
end
if any(strcmp(manifest.Properties.VariableNames, 'timeseries_path'))
    keep_mask = ~ismissing(manifest.timeseries_path) & strlength(strtrim(manifest.timeseries_path)) > 0;
    skipped_rows = sum(~keep_mask);
    if skipped_rows > 0
        fprintf('  Skipping %d rows without timeseries_path after extraction QC.\n', skipped_rows);
        manifest = manifest(keep_mask, :);
    end
end
fprintf('  Rows to process: %d\n', height(manifest));

fprintf('Loading ROI-to-network mapping...\n');
roi_network_map = table();
if isfile(ROI_NETWORK_MAP_PATH)
    roi_network_map = readtable(ROI_NETWORK_MAP_PATH, 'TextType', 'string', 'VariableNamingRule', 'preserve');
else
    fprintf('  !! ROI network map not found. State-network correlations will be skipped.\n');
end

energy_tbl = table();
dyn_tbl = table();
disp_tbl = table();
state_net_tbl = table();
centroid_tbl = table();
template_tbl = table();

empty_cache = struct( ...
    'Dataset', "", ...
    'SubjectID', "", ...
    'Session', "", ...
    'Condition', "", ...
    'RunLabel', "", ...
    'FileKey', "", ...
    'MeanFD', NaN, ...
    'NumFrames', NaN, ...
    'NumROIs', NaN, ...
    'HasNetworkMap', false, ...
    'NetID', categorical(), ...
    'ValidNetworks', {{}}, ...
    'V', [] ...
);
run_cache = repmat(empty_cache, height(manifest), 1);

% -------------------------------------------------------------------------
% Pass 1: per-run energy metrics and leading-eigenvector windows.
% -------------------------------------------------------------------------
for i = 1:height(manifest)
    row = manifest(i, :);
    ts_path = char(row.timeseries_path);
    fprintf('[%d/%d] %s %s %s %s\n', i, height(manifest), row.dataset, row.subject_id, row.session, row.condition);

    if ~isfile(ts_path)
        fprintf('  !! Missing timeseries file: %s\n', ts_path);
        continue;
    end

    ts_mat = load(ts_path);
    if ~isfield(ts_mat, 'time_series')
        fprintf('  !! Missing time_series in %s\n', ts_path);
        continue;
    end

    ts_TxR = double(ts_mat.time_series);
    if ndims(ts_TxR) ~= 2
        fprintf('  !! Unexpected shape in %s: [%d %d]\n', ts_path, size(ts_TxR, 1), size(ts_TxR, 2));
        continue;
    end
    ts_RxT = ts_TxR';
    n_rois = size(ts_TxR, 2);
    [has_network_map, net_id, valid_networks] = infer_network_map(roi_network_map, n_rois);

    mean_fd = coerce_numeric_scalar(row.mean_fd);
    file_key = string(sprintf('%s|%s|%s|%s', row.dataset, row.subject_id, row.session, row.run_label));

    energy_row = table();
    energy_row.Dataset = string(row.dataset);
    energy_row.SubjectID = string(row.subject_id);
    energy_row.Session = string(row.session);
    energy_row.Condition = string(row.condition);
    energy_row.RunLabel = string(row.run_label);
    energy_row.FileKey = file_key;
    energy_row.MeanFD = mean_fd;
    energy_row.NumFrames = size(ts_TxR, 1);
    energy_row.NumROIs = n_rois;
    energy_row.EnergyMean = NaN;
    energy_row.EnergySD = NaN;
    energy_row.StationaryMassDeepWells = NaN;
    energy_row.StationaryMassHighPeaks = NaN;
    energy_row.WellSpeed = NaN;
    energy_row.HillSpeed = NaN;
    energy_row.RadiusGyration = NaN;
    energy_row.QED = NaN;

    V = [];
    try
        [Phi_cortex, qed, P] = calculate_qdm_full(ts_RxT, QDM_PARAMS);
        energy_row.QED = qed;
        pi_stat = stationary_distribution(P);
        energy = -log(pi_stat + eps);
        energy_row.EnergyMean = mean(energy, 'omitnan');
        energy_row.EnergySD = std(energy, 'omitnan');
        energy_row.StationaryMassDeepWells = sum(pi_stat(energy <= prctile(energy, 10)), 'omitnan');
        energy_row.StationaryMassHighPeaks = sum(pi_stat(energy >= prctile(energy, 90)), 'omitnan');

        speeds = sqrt(sum(diff(Phi_cortex, 1, 1) .^ 2, 2));
        energy_prev = energy(1:end-1);
        if ~isempty(speeds)
            energy_row.WellSpeed = mean(speeds(energy_prev <= prctile(energy_prev, 10)), 'omitnan');
            energy_row.HillSpeed = mean(speeds(energy_prev >= prctile(energy_prev, 90)), 'omitnan');
        end

        centroid = mean(Phi_cortex, 1, 'omitnan');
        energy_row.RadiusGyration = mean(sqrt(sum((Phi_cortex - centroid) .^ 2, 2)), 'omitnan');
    catch ME
        fprintf('  !! Energy/QDM failed: %s\n', ME.message);
    end
    energy_tbl = [energy_tbl; energy_row]; %#ok<AGROW>

    try
        V = leading_eigenvector_windows(ts_TxR, STATE_WLEN, STATE_STEP);
    catch ME
        fprintf('  !! Leading-eigenvector extraction failed: %s\n', ME.message);
        V = [];
    end

    run_cache(i).Dataset = string(row.dataset);
    run_cache(i).SubjectID = string(row.subject_id);
    run_cache(i).Session = string(row.session);
    run_cache(i).Condition = string(row.condition);
    run_cache(i).RunLabel = string(row.run_label);
    run_cache(i).FileKey = file_key;
    run_cache(i).MeanFD = mean_fd;
    run_cache(i).NumFrames = size(ts_TxR, 1);
    run_cache(i).NumROIs = n_rois;
    run_cache(i).HasNetworkMap = has_network_map;
    run_cache(i).NetID = net_id;
    run_cache(i).ValidNetworks = valid_networks;
    run_cache(i).V = V;
end

% -------------------------------------------------------------------------
% Fit shared state templates within each dataset.
% -------------------------------------------------------------------------
template_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
datasets = unique(string({run_cache.Dataset}));
datasets = datasets(strlength(datasets) > 0);

for d = 1:numel(datasets)
    dataset_name = datasets(d);
    pooled = [];
    contributing_runs = 0;
    for i = 1:numel(run_cache)
        if run_cache(i).Dataset ~= dataset_name
            continue;
        end
        V = run_cache(i).V;
        if isempty(V) || size(V, 1) < STATE_K
            continue;
        end
        keep_n = min(size(V, 1), STATE_MAX_WINDOWS_PER_RUN);
        keep_idx = randperm(size(V, 1), keep_n);
        pooled = [pooled; V(keep_idx, :)]; %#ok<AGROW>
        contributing_runs = contributing_runs + 1;
    end

    if size(pooled, 1) < STATE_K
        fprintf('  !! Skipping shared states for %s: only %d pooled windows\n', dataset_name, size(pooled, 1));
        continue;
    end

    fprintf('Fitting shared states for %s using %d pooled windows from %d runs\n', dataset_name, size(pooled, 1), contributing_runs);
    try
        [pooled_idx, C] = kmeans(pooled, STATE_K, ...
            'Replicates', 20, ...
            'MaxIter', 500, ...
            'OnlinePhase', 'off', ...
            'EmptyAction', 'singleton');
    catch ME
        fprintf('  !! Shared-state k-means failed for %s: %s\n', dataset_name, ME.message);
        continue;
    end

    occ = accumarray(pooled_idx(:), 1, [STATE_K, 1]) / size(pooled, 1);
    [occ_sorted, order] = sort(occ, 'descend');
    C = C(order, :);
    template_map(char(dataset_name)) = C;

    for k = 1:STATE_K
        t_row = table();
        t_row.Dataset = dataset_name;
        t_row.StateID = k;
        t_row.PooledOccupancy = occ_sorted(k);
        t_row.NSampledWindows = sum(pooled_idx == order(k));
        t_row.NTemplateROIs = size(C, 2);
        for roi = 1:size(C, 2)
            t_row.(sprintf('ROI_%03d', roi)) = C(k, roi);
        end
        template_tbl = [template_tbl; t_row]; %#ok<AGROW>
    end
end

% -------------------------------------------------------------------------
% Pass 2: assign each run to shared templates and compute valid
% state-specific summaries.
% -------------------------------------------------------------------------
for i = 1:numel(run_cache)
    rc = run_cache(i);
    if strlength(rc.Dataset) == 0 || isempty(rc.V)
        continue;
    end
    if ~isKey(template_map, char(rc.Dataset))
        continue;
    end

    shared_C = template_map(char(rc.Dataset));
    if size(shared_C, 2) ~= size(rc.V, 2)
        fprintf('  !! Shared template ROI mismatch for %s %s\n', rc.Dataset, rc.FileKey);
        continue;
    end

    D = pdist2(rc.V, shared_C, 'euclidean');
    [~, idx_assign] = min(D, [], 2);
    [occupancy, switch_rate, entH] = state_dynamics(idx_assign, STATE_K);

    dyn_row = table();
    dyn_row.Dataset = rc.Dataset;
    dyn_row.SubjectID = rc.SubjectID;
    dyn_row.Session = rc.Session;
    dyn_row.Condition = rc.Condition;
    dyn_row.RunLabel = rc.RunLabel;
    dyn_row.FileKey = rc.FileKey;
    dyn_row.MeanFD = rc.MeanFD;
    dyn_row.NumStates = STATE_K;
    dyn_row.SwitchRate = switch_rate;
    dyn_row.Entropy = entH;
    for s = 1:STATE_K
        dyn_row.(sprintf('Occupancy_State%d', s)) = occupancy(s);
    end
    dyn_tbl = [dyn_tbl; dyn_row]; %#ok<AGROW>

    run_centroids = [];
    for k = 1:STATE_K
        mask = idx_assign == k;
        if ~any(mask)
            continue;
        end

        centroid_vec = mean(rc.V(mask, :), 1, 'omitnan');
        run_centroids = [run_centroids; centroid_vec]; %#ok<AGROW>

        c_row = table();
        c_row.Dataset = rc.Dataset;
        c_row.SubjectID = rc.SubjectID;
        c_row.Session = rc.Session;
        c_row.Condition = rc.Condition;
        c_row.RunLabel = rc.RunLabel;
        c_row.FileKey = rc.FileKey;
        c_row.StateID = k;
        c_row.MeanFD = rc.MeanFD;
        c_row.StateOccupancy = occupancy(k);
        c_row.StateWindows = sum(mask);
        for roi = 1:numel(centroid_vec)
            c_row.(sprintf('ROI_%03d', roi)) = centroid_vec(roi);
        end
        centroid_tbl = [centroid_tbl; c_row]; %#ok<AGROW>

        if rc.HasNetworkMap
            state_vec = centroid_vec(:);
            for r = 1:numel(rc.ValidNetworks)
                net_name = char(rc.ValidNetworks{r});
                template = double(rc.NetID == net_name);
                sn_row = table();
                sn_row.Dataset = rc.Dataset;
                sn_row.SubjectID = rc.SubjectID;
                sn_row.Session = rc.Session;
                sn_row.Condition = rc.Condition;
                sn_row.RunLabel = rc.RunLabel;
                sn_row.FileKey = rc.FileKey;
                sn_row.StateID = k;
                sn_row.Network = string(net_name);
                sn_row.Correlation = corr(state_vec, template, 'rows', 'pairwise');
                sn_row.MeanFD = rc.MeanFD;
                state_net_tbl = [state_net_tbl; sn_row]; %#ok<AGROW>
            end
        end
    end

    disp_row = table();
    disp_row.Dataset = rc.Dataset;
    disp_row.SubjectID = rc.SubjectID;
    disp_row.Session = rc.Session;
    disp_row.Condition = rc.Condition;
    disp_row.RunLabel = rc.RunLabel;
    disp_row.FileKey = rc.FileKey;
    disp_row.MeanFD = rc.MeanFD;
    disp_row.NumCentroids = size(run_centroids, 1);
    disp_row.RawDispersion = NaN;
    disp_row.PCADispersion = NaN;
    disp_row.PC1_VarExplained = NaN;
    disp_row.PC2_VarExplained = NaN;
    disp_row.PC3_VarExplained = NaN;
    if size(run_centroids, 1) >= 2
        disp_row.RawDispersion = mean(pdist(run_centroids), 'omitnan');
    end
    if size(run_centroids, 1) >= 4
        [~, score, latent] = pca(run_centroids);
        max_pc = min(3, size(score, 2));
        if max_pc >= 2
            disp_row.PCADispersion = mean(pdist(score(:, 1:max_pc)), 'omitnan');
        end
        var_explained = 100 * latent ./ sum(latent);
        if numel(var_explained) >= 1, disp_row.PC1_VarExplained = var_explained(1); end
        if numel(var_explained) >= 2, disp_row.PC2_VarExplained = var_explained(2); end
        if numel(var_explained) >= 3, disp_row.PC3_VarExplained = var_explained(3); end
    end
    disp_tbl = [disp_tbl; disp_row]; %#ok<AGROW>
end

write_csv_safe(template_tbl, fullfile(OUTDIR, 'harmonized_shared_state_templates.csv'));
write_csv_safe(dyn_tbl, fullfile(OUTDIR, 'harmonized_state_dynamics_run_metrics.csv'));
write_csv_safe(disp_tbl, fullfile(OUTDIR, 'harmonized_centroid_dispersion_by_run.csv'));
write_csv_safe(energy_tbl, fullfile(OUTDIR, 'harmonized_energy_landscape_metrics.csv'));
write_csv_safe(state_net_tbl, fullfile(OUTDIR, 'harmonized_state_network_correlations.csv'));
write_csv_safe(centroid_tbl, fullfile(OUTDIR, 'harmonized_state_centroids.csv'));

save(fullfile(OUTDIR, 'harmonized_extended_metrics.mat'), ...
    'template_tbl', 'dyn_tbl', 'disp_tbl', 'energy_tbl', 'state_net_tbl', 'centroid_tbl', ...
    'QDM_PARAMS', 'STATE_WLEN', 'STATE_STEP', 'STATE_K', 'STATE_MAX_WINDOWS_PER_RUN', '-v7.3');

fprintf('Done. Outputs written to %s\n', OUTDIR);

function [has_network_map, net_id, valid_networks] = infer_network_map(roi_network_map, n_rois)
    if ~isempty(roi_network_map) && height(roi_network_map) == n_rois
        has_network_map = true;
        net_id = categorical(string(roi_network_map.Network));
        valid_networks = categories(net_id(~isundefined(net_id) & string(net_id) ~= "none"));
    else
        has_network_map = false;
        net_id = categorical();
        valid_networks = {};
    end
end

function value = coerce_numeric_scalar(x)
    if isnumeric(x)
        value = double(x);
        return;
    end
    if iscell(x)
        x = x{1};
    end
    if ismissing(x) || strlength(string(x)) == 0
        value = NaN;
        return;
    end
    value = str2double(string(x));
    if isnan(value)
        value = NaN;
    end
end

function [Phi_quantum, q_eff_dim, P] = calculate_qdm_full(ts_subj, qdm_params)
    ts_subj = detrend(ts_subj', 'constant')';
    ts_subj = zscore(ts_subj, [], 2);
    ts_subj(isnan(ts_subj)) = 0;
    Tm_subj = size(ts_subj, 2);

    D_pairwise_sq = squareform(pdist(ts_subj')).^2;
    try
        Kq = exp(1i * D_pairwise_sq / qdm_params.epsilon_quantum);
        Pq_intermediate = Kq ^ qdm_params.Thorizont_quantum;
    catch
        [Vk, Lk] = eig(exp(1i * D_pairwise_sq / qdm_params.epsilon_quantum));
        Pq_intermediate = Vk * (Lk .^ qdm_params.Thorizont_quantum) / Vk;
    end

    P_abs2 = abs(Pq_intermediate) .^ 2;
    d = sum(P_abs2, 2);
    d(d < 1e-9) = 1;
    P = diag(1 ./ d) * P_abs2;

    [V_raw, L_raw] = eig(P);
    lam = real(diag(L_raw));
    [lam_sorted, idx] = sort(lam, 'descend');
    V = V_raw(:, idx);
    L_sorted = diag(lam_sorted);

    actual_latdim = min(qdm_params.LATDIM, find(lam_sorted < 0.001, 1) - 2);
    if isempty(actual_latdim) || actual_latdim < 1
        actual_latdim = qdm_params.LATDIM;
    end
    actual_latdim = min(actual_latdim, Tm_subj - 2);
    actual_latdim = max(1, actual_latdim);

    Phi_quantum = V(:, 2:actual_latdim + 1) * ...
        sqrt(abs(L_sorted(2:actual_latdim + 1, 2:actual_latdim + 1)));
    C = cov(bsxfun(@minus, Phi_quantum, mean(Phi_quantum, 1)));
    ev = eig(C);
    ev = ev(isfinite(ev) & ev > 1e-9);
    q_eff_dim = (sum(ev) ^ 2) / sum(ev .^ 2);
end

function pi_stat = stationary_distribution(P)
    [V, D] = eig(P');
    [~, idx] = max(real(diag(D)));
    pi_stat = abs(real(V(:, idx)));
    pi_stat = pi_stat ./ sum(pi_stat);
end

function V = leading_eigenvector_windows(TS, wlen, step)
    [T, N] = size(TS);
    idx = 1:step:(T - wlen + 1);
    V = zeros(numel(idx), N);
    c = 1;
    for s = idx
        X = TS(s:s + wlen - 1, :);
        C = corrcoef(X, 'Rows', 'pairwise');
        C(isnan(C)) = 0;
        C = (C + C') / 2;
        try
            [vec, ~] = eigs(C, 1, 'largestreal');
        catch
            [Vfull, Dfull] = eig(C);
            [~, idx_max] = max(real(diag(Dfull)));
            vec = Vfull(:, idx_max);
        end
        if sum(vec, 'omitnan') < 0
            vec = -vec;
        end
        V(c, :) = vec(:)';
        c = c + 1;
    end
end

function [occupancy, switch_rate, entH] = state_dynamics(idx, K)
    if nargin < 2
        K = max(idx);
    end
    T = numel(idx);
    occupancy = accumarray(idx(:), 1, [K, 1]) / T;
    switch_rate = mean(abs(diff(idx)) > 0);
    p = occupancy(occupancy > 0);
    entH = -sum(p .* log2(p));
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

function ensure_dir(pathstr)
    if ~exist(pathstr, 'dir')
        mkdir(pathstr);
    end
end

function value = getenv_default(name, fallback)
    value = getenv(name);
    if isempty(value)
        value = fallback;
    end
end
