% ----------------------------------------------------------------------------
% Structured Semi-Markov Decoding with Conformal Prediction for
% Wrist-Actigraphy Sleep Event Detection
%
% Author    : Hoang Nhut Huynh
% Copyright : (c) 2026 Hoang Nhut Huynh
% ----------------------------------------------------------------------------
%% Fig 6: Bland–Altman Plots for Newcastle PSG
% Reads: outputs/newcastle/results/per_recording.csv
% Output: fig6_bland_altman.pdf
%
% Bland–Altman: x = mean(pred, true), y = pred - true
% Shows onset (left) and wakeup (right) panels

clear; close all; clc;

%% ─── Config ───────────────────────────────────────────────────────────
base_dir = fileparts(fileparts(mfilename('fullpath')));  % repository root
csv_path = fullfile(base_dir, 'outputs', 'newcastle', 'results', 'per_recording.csv');

%% ─── Read ────────────────────────────────────────────────────────────
T = readtable(csv_path, 'TextType', 'string');

% per_recording.csv columns include:
%   onset_err_min, wake_err_min, spt_pred_min, spt_true_min, psg_spt_min
% onset_err_min = predicted_onset - true_onset (in minutes, signed)
% wake_err_min  = predicted_wake  - true_wake  (in minutes, signed)

% Filter only observable events
if ismember('onset_observable', T.Properties.VariableNames)
    onset_mask = strcmpi(string(T.onset_observable), "True");
    wake_mask  = strcmpi(string(T.wake_observable), "True");
else
    onset_mask = true(height(T), 1);
    wake_mask  = true(height(T), 1);
end

onset_err = T.onset_err_min(onset_mask);
wake_err  = T.wake_err_min(wake_mask);

% For Bland-Altman we need both pred and true.
% onset_err = pred - true → pred = true + err
% We need SPT or clock times. Since we only have errors,
% use error directly (y-axis) and approximate mean.
% Alternative: reconstruct from spt_pred and spt_true for SPT BA.

spt_pred = T.spt_pred_min;
spt_true = T.spt_true_min;
spt_diff = spt_pred - spt_true;
spt_mean = (spt_pred + spt_true) / 2;

%% ─── BA statistics ──────────────────────────────────────────────────
% SPT Bland-Altman
ba_bias = mean(spt_diff);
ba_sd   = std(spt_diff);
ba_loa_lo = ba_bias - 1.96 * ba_sd;
ba_loa_hi = ba_bias + 1.96 * ba_sd;

% Proportional bias regression
if numel(spt_mean) > 2
    p_reg = polyfit(spt_mean, spt_diff, 1);
    x_fit = linspace(min(spt_mean), max(spt_mean), 100);
    y_fit = polyval(p_reg, x_fit);
end

%% ─── Figure: 3-panel (onset err, wake err, SPT BA) ──────────────────
c_dot  = [0.200, 0.467, 0.800];
c_bias = [0.850, 0.325, 0.098];
c_loa  = [0.500, 0.500, 0.500];
c_reg  = [0.580, 0.000, 0.827];

fig = figure('Units','centimeters','Position',[2 2 18 12]);

% ── Panel (a): Onset error distribution ──
ax1 = subplot(2,2,1); hold on; box on; grid on;
if numel(onset_err) > 1
    histogram(onset_err, 'FaceColor', c_dot, 'EdgeColor', 'w', ...
        'FaceAlpha', 0.7);
end
scatter(onset_err, 1:numel(onset_err), 50, c_dot, 'filled', 'MarkerFaceAlpha', 0.7);
xlabel('Onset error (min)', 'FontSize', 9);
ylabel('Recording index', 'FontSize', 9);
title('(a) Onset error', 'FontSize', 10);
xline(0, '--k', 'LineWidth', 0.8);
xline(30, ':r', 'LineWidth', 0.8, 'Label', '30 min');
xline(-30, ':r', 'LineWidth', 0.8);
set(ax1, 'FontSize', 8, 'TickDir', 'out');

% ── Panel (b): Wake error distribution ──
ax2 = subplot(2,2,2); hold on; box on; grid on;
scatter(wake_err, 1:numel(wake_err), 50, c_dot, 'filled', 'MarkerFaceAlpha', 0.7);
xlabel('Wakeup error (min)', 'FontSize', 9);
ylabel('Recording index', 'FontSize', 9);
title('(b) Wakeup error', 'FontSize', 10);
xline(0, '--k', 'LineWidth', 0.8);
xline(30, ':r', 'LineWidth', 0.8);
xline(-30, ':r', 'LineWidth', 0.8);
set(ax2, 'FontSize', 8, 'TickDir', 'out');

% ── Panel (c): SPT Bland–Altman ──
ax3 = subplot(2,2,[3 4]); hold on; box on; grid on;

scatter(spt_mean, spt_diff, 60, c_dot, 'filled', 'MarkerFaceAlpha', 0.7);

% Bias line
yline(ba_bias, '-', 'Color', c_bias, 'LineWidth', 1.5, ...
    'Label', sprintf('Bias = %.1f min', ba_bias), ...
    'LabelHorizontalAlignment', 'left', 'FontSize', 7);

% LoA lines
yline(ba_loa_lo, '--', 'Color', c_loa, 'LineWidth', 1, ...
    'Label', sprintf('-1.96 SD = %.1f', ba_loa_lo), ...
    'LabelHorizontalAlignment', 'left', 'FontSize', 7);
yline(ba_loa_hi, '--', 'Color', c_loa, 'LineWidth', 1, ...
    'Label', sprintf('+1.96 SD = %.1f', ba_loa_hi), ...
    'LabelHorizontalAlignment', 'left', 'FontSize', 7);

% Proportional bias regression
if exist('x_fit', 'var')
    plot(x_fit, y_fit, '-', 'Color', c_reg, 'LineWidth', 1.5);
    text(x_fit(end), y_fit(end), sprintf(' slope=%.2f', p_reg(1)), ...
        'Color', c_reg, 'FontSize', 7, 'VerticalAlignment', 'bottom');
end

xlabel('Mean of predicted and true SPT (min)', 'FontSize', 9);
ylabel('Predicted - True SPT (min)', 'FontSize', 9);
title('(c) Bland–Altman: Sleep Period Time', 'FontSize', 10);
set(ax3, 'FontSize', 8, 'TickDir', 'out');
% yline does not expand the axes: force limits so both LoA lines (+618/-311) are visible
y_lo = min([ba_loa_lo; spt_diff]); y_hi = max([ba_loa_hi; spt_diff]);
ylim(ax3, [y_lo, y_hi] + [-0.08, 0.12] * (y_hi - y_lo));

%% ─── Export ──────────────────────────────────────────────────────────
% exportgraphics crops to the figure content (print -dpdf writes a full letter page)
exportgraphics(fig, fullfile(base_dir, 'outputs', 'fig6_bland_altman.pdf'), 'ContentType', 'vector');
exportgraphics(fig, fullfile(base_dir, 'outputs', 'fig6_bland_altman.png'), 'Resolution', 300);
fprintf('Saved fig6_bland_altman.pdf/.png\n');
