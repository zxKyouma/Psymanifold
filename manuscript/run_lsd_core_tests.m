input_path = getenv('LSD_CORE_METRICS_CSV');
output_path = getenv('LSD_CORE_TESTS_CSV');
if strlength(string(input_path)) == 0
    input_path = fullfile(pwd, 'outputs', 'harmonized_psychedelics_summary', 'harmonized_metric_subject_condition_means.csv');
end
if strlength(string(output_path)) == 0
    output_path = fullfile(pwd, 'outputs', 'pharmacological_specificity', 'lsd_core_metric_tests.csv');
end

T = readtable(input_path);
if ismember('Dataset', T.Properties.VariableNames)
    T = T(strcmp(string(T.Dataset), "LSD"), :);
end
T = T(ismember(string(T.Condition), ["Placebo","Psychedelic"]), :);
T.SubjectID = categorical(string(T.SubjectID));
T.Condition = categorical(string(T.Condition));
metrics = {'Q','PC','FC_within','FC_between','SwitchRate','Entropy','QED','NGSC','NGSC_vtx'};
R = table();
for i = 1:numel(metrics)
    metric = metrics{i};
    if ~ismember(metric, T.Properties.VariableNames), continue; end
    vals = T.(metric);
    if all(isnan(vals)), continue; end
    D = T(:, {'SubjectID','Condition',metric});
    D = rmmissing(D);
    if height(D) < 4, continue; end
    G = groupsummary(D, {'SubjectID','Condition'}, 'mean', metric);
    P = unstack(G, ['mean_' metric], 'Condition');
    row = table(string(metric), 'VariableNames', {'Metric'});
    if all(ismember({'Placebo','Psychedelic'}, P.Properties.VariableNames))
        diffs = P.Psychedelic - P.Placebo;
        diffs = diffs(isfinite(diffs));
        if numel(diffs) >= 3
            [~, ptt, ~, st] = ttest(diffs);
            row.NPaired = numel(diffs);
            row.PlaceboMean = mean(P.Placebo, 'omitnan');
            row.PsychedelicMean = mean(P.Psychedelic, 'omitnan');
            row.DeltaMean = mean(diffs, 'omitnan');
            row.TPaired = st.tstat;
            row.PPaired = ptt;
        else
            row.NPaired = numel(diffs); row.PlaceboMean = mean(P.Placebo, 'omitnan'); row.PsychedelicMean = mean(P.Psychedelic, 'omitnan'); row.DeltaMean = mean(diffs, 'omitnan'); row.TPaired = NaN; row.PPaired = NaN;
        end
    else
        row.NPaired = 0; row.PlaceboMean = NaN; row.PsychedelicMean = NaN; row.DeltaMean = NaN; row.TPaired = NaN; row.PPaired = NaN;
    end
    try
        lme = fitlme(D, sprintf('%s ~ Condition + (1|SubjectID)', metric));
        [~,~,stats] = fixedEffects(lme);
        idx = find(strcmp(stats.Name, 'Condition_Psychedelic'), 1);
        if isempty(idx), idx = 2; end
        row.NLME = height(D);
        row.EstimateLME = stats.Estimate(idx);
        row.SELME = stats.SE(idx);
        row.TLME = stats.tStat(idx);
        row.PLME = stats.pValue(idx);
    catch
        row.NLME = height(D); row.EstimateLME = NaN; row.SELME = NaN; row.TLME = NaN; row.PLME = NaN;
    end
    R = [R; row];
end
writetable(R, output_path);
