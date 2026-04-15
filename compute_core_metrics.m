% =========================================================================
% HARMONIZED CORE METRIC RUNNER
%
% Runs the shared multiscale metric stack on the completed harmonized
% Gordon333 mean ROI time-series manifest.
%
% Note on terminology:
% The state-dynamics layer uses leading eigenvectors of sliding-window
% correlation matrices. This is a leading-eigenvector state-dynamics method,
% not classical phase-coherence LEiDA. The entropy summary is the Shannon
% entropy of cluster occupancies, not a temporal entropy-rate estimate.
% =========================================================================

clear; clc; close all; rng(42);

fprintf('--- Harmonized Core Metric Runner ---\n');

ROOT = fileparts(mfilename('fullpath'));
MANIFEST_PATH = getenv_default('HARMONIZED_TS_MANIFEST', fullfile(ROOT, 'outputs', 'harmonized_psychedelics', 'harmonized_timeseries_manifest.tsv'));
ROI_NETWORK_MAP_PATH = getenv_default('ROI_NETWORK_MAP_PATH', fullfile(ROOT, 'assets', 'gordon_roi_network_mapping.csv'));
OUTDIR = getenv_default('HARMONIZED_METRIC_OUTDIR', fullfile(ROOT, 'outputs', 'harmonized_psychedelics_core_metrics'));
MAX_ROWS = str2double(getenv_default('HARMONIZED_MAX_ROWS', '0'));
ensure_dir(OUTDIR);

QDM_PARAMS.LATDIM = 18;
QDM_PARAMS.Thorizont_quantum = 2;
QDM_PARAMS.epsilon_quantum = 70;
STATE_WLEN = 30;
STATE_STEP = 1;
STATE_K = 6;
PROCRUSTES_K = 5;

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

fprintf('Loading Gordon ROI-to-network mapping...\n');
roi_network_map = table();
net_id = categorical();
valid_networks = {};
has_network_map = false;
if isfile(ROI_NETWORK_MAP_PATH)
    roi_network_map = readtable(ROI_NETWORK_MAP_PATH, 'TextType', 'string', 'VariableNamingRule', 'preserve');
else
    fprintf('  !! ROI network map not found. Network-dependent metrics will be skipped.\n');
end

results = table();
netproc_tbl = table();
netglobal_tbl = table();

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
        fprintf('  !! Unexpected shape in %s: [%d %d]\n', ts_path, size(ts_TxR,1), size(ts_TxR,2));
        continue;
    end
    ts_RxT = ts_TxR';
    n_rois = size(ts_TxR, 2);

    if ~isempty(roi_network_map) && height(roi_network_map) == n_rois
        has_network_map = true;
        net_id = categorical(string(roi_network_map.Network));
        valid_networks = categories(net_id(~isundefined(net_id) & string(net_id) ~= "none"));
    else
        has_network_map = false;
        net_id = categorical();
        valid_networks = {};
    end

    fc = compute_fc_from_timeseries(ts_TxR);
    mean_fd = coerce_numeric_scalar(row.mean_fd);

    out_row = table();
    out_row.Dataset = string(row.dataset);
    out_row.SubjectID = string(row.subject_id);
    out_row.Session = string(row.session);
    out_row.Condition = string(row.condition);
    out_row.RunLabel = string(row.run_label);
    out_row.FileKey = string(sprintf('%s|%s|%s|%s', row.dataset, row.subject_id, row.session, row.run_label));
    out_row.TimeSeriesPath = string(ts_path);
    out_row.InputPath = string(row.input_path);
    out_row.NumFrames = size(ts_TxR, 1);
    out_row.NumROIs = size(ts_TxR, 2);
    out_row.MeanFD = mean_fd;

    try
        [Phi_cortex, qed] = calculate_qdm_manifold_and_id(ts_RxT, QDM_PARAMS);
    catch ME
        fprintf('  !! QDM failed: %s\n', ME.message);
        Phi_cortex = [];
        qed = NaN;
    end
    out_row.QED = qed;

    out_row.NGSC = compute_ngsc(ts_RxT);
    out_row.Q = NaN;
    out_row.PC = NaN;
    out_row.FC_within = NaN;
    out_row.FC_between = NaN;
    if has_network_map
        try
            % Use a positive-edge convention for FC summaries.
            [fc_within, fc_between] = within_between_FC(max(fc, 0), net_id);
            out_row.FC_within = fc_within;
            out_row.FC_between = fc_between;
        catch ME
            fprintf('  !! within/between FC failed: %s\n', ME.message);
        end
        try
            [q_val, pc_val] = compute_topology_metrics_from_fc(fc, net_id);
            out_row.Q = q_val;
            out_row.PC = pc_val;
        catch ME
            fprintf('  !! Q/PC failed: %s\n', ME.message);
        end
    end

    % Canonical state-dynamics metrics are derived from dataset-specific
    % shared templates in compute_extended_metrics.m. Leave placeholders
    % here so downstream consumers do not silently inherit run-local values.
    out_row.SwitchRate = NaN;
    out_row.Entropy = NaN;

    results = [results; out_row]; %#ok<AGROW>

    if has_network_map
        net_phi_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
        for n = 1:numel(valid_networks)
            net_name = char(valid_networks{n});
            idx_net = net_id == net_name;
            if nnz(idx_net) < 2
                continue;
            end
            try
                [phi_net, ~] = calculate_qdm_manifold_and_id(ts_RxT(idx_net, :), QDM_PARAMS);
                net_phi_map(net_name) = phi_net;
            catch
            end
        end

        net_names = sort(string(keys(net_phi_map)));
        for n = 1:numel(net_names)
            net_name = char(net_names(n));
            try
                dg = procrustes_dist(Phi_cortex, net_phi_map(net_name), PROCRUSTES_K);
                ng_row = table();
                ng_row.Dataset = string(row.dataset);
                ng_row.SubjectID = string(row.subject_id);
                ng_row.Session = string(row.session);
                ng_row.Condition = string(row.condition);
                ng_row.RunLabel = string(row.run_label);
                ng_row.FileKey = out_row.FileKey;
                ng_row.Network = string(net_name);
                ng_row.ProcrustesGlobalDist = dg;
                ng_row.MeanFD = mean_fd;
                netglobal_tbl = [netglobal_tbl; ng_row]; %#ok<AGROW>
            catch
            end
        end

        for a = 1:numel(net_names)
            for b = a + 1:numel(net_names)
                net_a = char(net_names(a));
                net_b = char(net_names(b));
                d = procrustes_dist(net_phi_map(net_a), net_phi_map(net_b), PROCRUSTES_K);
                pair_row = table();
                pair_row.Dataset = string(row.dataset);
                pair_row.SubjectID = string(row.subject_id);
                pair_row.Session = string(row.session);
                pair_row.Condition = string(row.condition);
                pair_row.RunLabel = string(row.run_label);
                pair_row.FileKey = out_row.FileKey;
                pair_row.Pair = string(sprintf('%s-%s', net_a, net_b));
                pair_row.ProcrustesNetDist = d;
                pair_row.MeanFD = mean_fd;
                netproc_tbl = [netproc_tbl; pair_row]; %#ok<AGROW>
            end
        end
    end
end

write_csv_safe(results, fullfile(OUTDIR, 'harmonized_results_summary.csv'));
write_csv_safe(netproc_tbl, fullfile(OUTDIR, 'harmonized_network_procrustes_distances.csv'));
write_csv_safe(netglobal_tbl, fullfile(OUTDIR, 'harmonized_network_to_global_distances.csv'));

save(fullfile(OUTDIR, 'harmonized_core_metrics.mat'), ...
    'results', 'netproc_tbl', 'netglobal_tbl', ...
    'QDM_PARAMS', 'STATE_WLEN', 'STATE_STEP', 'STATE_K', 'PROCRUSTES_K', '-v7.3');

fprintf('Done. Outputs written to %s\n', OUTDIR);

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

function fc = compute_fc_from_timeseries(ts_TxR)
    fc = corr(ts_TxR, 'Rows', 'pairwise');
    fc(~isfinite(fc)) = 0;
    fc = (fc + fc') / 2;
end

function [Phi_quantum, q_eff_dim] = calculate_qdm_manifold_and_id(ts_subj, qdm_params)
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

function d = procrustes_dist(X, Y, k_keep)
    d = NaN;
    if isempty(X) || isempty(Y)
        return;
    end

    T_common = min(size(X, 1), size(Y, 1));
    X = interp1(linspace(0, 1, size(X, 1)), X, linspace(0, 1, T_common));
    Y = interp1(linspace(0, 1, size(Y, 1)), Y, linspace(0, 1, T_common));

    k = min([size(X, 2), size(Y, 2), k_keep]);
    if k < 1
        return;
    end

    XA = zscore(X(:, 1:k), 0, 1);
    XB = zscore(Y(:, 1:k), 0, 1);
    thr = 1e-15;
    keepA = any(var(XA, 0, 1) > thr) & all(isfinite(XA), 1);
    keepB = any(var(XB, 0, 1) > thr) & all(isfinite(XB), 1);
    XA = XA(:, keepA);
    XB = XB(:, keepB);
    k_final = min(size(XA, 2), size(XB, 2));

    if k_final < 1
        return;
    end

    try
        [d, ~, ~] = procrustes(XA(:, 1:k_final), XB(:, 1:k_final), 'reflection', false);
    catch
        d = NaN;
    end
end

function [FCw, FCb] = within_between_FC(A, net_id)
    valid_mask = ~isundefined(net_id) & string(net_id) ~= "none";
    nets = categories(net_id(valid_mask));
    m = numel(nets);
    W = NaN(m, 1);
    B = NaN(m, 1);

    for i = 1:m
        mask_i = net_id == nets{i};
        if nnz(mask_i) < 2
            continue;
        end
        Ai = A(mask_i, mask_i);
        Ao = A(mask_i, valid_mask & ~mask_i);
        W(i) = mean(Ai(triu(true(size(Ai)), 1)), 'omitnan');
        B(i) = mean(Ao(:), 'omitnan');
    end

    FCw = mean(W, 'omitnan');
    FCb = mean(B, 'omitnan');
end

function [q_val, pc_mean] = compute_topology_metrics_from_fc(fc, net_id)
    A = double(fc);
    A(1:size(A, 1) + 1:end) = 0;
    A = (A + A') / 2;
    A(isnan(A)) = 0;
    Ap = max(A, 0);

    try
        [Ci, q_val] = modularity_und(Ap);
    catch
        [Ci, q_val] = fallback_modularity(Ap);
    end

    try
        PC_node = participation_coef(Ap, Ci);
    catch
        PC_node = fallback_participation(Ap, Ci);
    end
    pc_mean = mean(PC_node, 'omitnan');
end

function [Ci, Q] = fallback_modularity(A)
    n = size(A,1);
    d = sum(A,2);
    L = diag(d) - A;
    L = L + 1e-6 * speye(n);
    try
        [V,~] = eigs(L, 2, 'smallestreal');
    catch
        [V,~] = eig(full(L));
        V = V(:,1:2);
    end
    Ci = ones(n,1);
    Ci(V(:,2) > 0) = 2;
    Q = quality_modularity(A, Ci);
end

function PC = fallback_participation(A, Ci)
    n = size(A,1);
    K = max(Ci);
    ki = sum(A,2);
    PC = zeros(n,1);
    for i = 1:n
        if ki(i) == 0
            continue;
        end
        frac = zeros(K,1);
        for k = 1:K
            frac(k) = sum(A(i, Ci == k));
        end
        PC(i) = 1 - sum((frac ./ ki(i)) .^ 2);
    end
end

function Q = quality_modularity(A, Ci)
    m = sum(A(:)) / 2;
    ki = sum(A,2);
    Q = 0;
    for k = 1:max(Ci)
        nodes = find(Ci == k);
        ek = sum(sum(A(nodes, nodes))) / 2;
        ak = sum(ki(nodes)) / (2 * m);
        Q = Q + (ek / m - ak ^ 2);
    end
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

function [occupancy, switch_rate, entH] = state_dynamics(idx)
    K = max(idx);
    T = numel(idx);
    occupancy = accumarray(idx(:), 1, [K, 1]) / T;
    switch_rate = mean(abs(diff(idx)) > 0);
    p = occupancy + eps;
    entH = -sum(p .* log2(p));
end

function ngsc = compute_ngsc(ts_roiXt)
    ngsc = NaN;
    try
        if size(ts_roiXt, 2) < 2 || size(ts_roiXt, 1) < 2
            ngsc = 0;
            return;
        end
        X = detrend(ts_roiXt', 'constant')';
        X = zscore(X, [], 2);
        X(~isfinite(X)) = 0;
        [~, ~, latent] = pca(X', 'Centered', false);
        eigs_pos = latent(latent > 1e-9);
        k = numel(eigs_pos);
        if k < 2
            ngsc = 0;
            return;
        end
        p = eigs_pos ./ sum(eigs_pos);
        H = -sum(p .* log2(p));
        ngsc = H / log2(k);
    catch
        ngsc = NaN;
    end
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
