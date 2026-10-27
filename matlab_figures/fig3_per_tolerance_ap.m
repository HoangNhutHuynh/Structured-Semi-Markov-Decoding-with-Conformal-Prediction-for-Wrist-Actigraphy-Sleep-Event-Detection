% ----------------------------------------------------------------------------
% Structured Semi-Markov Decoding with Conformal Prediction for
% Wrist-Actigraphy Sleep Event Detection
%
% Author    : Hoang Nhut Huynh
% Copyright : (c) 2026 Hoang Nhut Huynh
% ----------------------------------------------------------------------------
%% Fig 3: Per-Tolerance AP Comparison (B1 vs P2)
% Reads: outputs/compare/per_tolerance_ap.csv
% Output: fig3_per_tolerance_ap.pdf
%
% NOTE: Container data is N=20 (small subset). On full data (7254 nights)
% the numbers will differ. Adjust base_dir to your path.

clear; close all; clc;

%% ─── Config ───────────────────────────────────────────────────────────
base_dir = fileparts(fileparts(mfilename('fullpath')));  % repository root
csv_path = fullfile(base_dir, 'outputs', 'compare', 'per_tolerance_ap.csv');

%% ─── Read CSV ─────────────────────────────────────────────────────────
% The CSV has a multi-row header. Parse manually.
fid = fopen(csv_path, 'r');
hdr1 = fgetl(fid);  % model row
hdr2 = fgetl(fid);  % mode row
fclose(fid);

% Read data portion (skip 2 header rows)
opts = detectImportOptions(csv_path, 'NumHeaderLines', 2);
T = readtable(csv_path, opts);

% Column layout: event, tolerance, B1_K1, B1_kaggle, P2_K1, P2_kaggle
% Adjust column names based on actual file
col_names = T.Properties.VariableNames;
event_col  = T{:, 1};   % 'onset' or 'wakeup'
tol_col    = T{:, 2};   % tolerance in steps

% Determine column indices for B1 kaggle and P2 K1
% Typically: col3=B1_K1, col4=B1_kaggle, col5=P2_K1, col6=P2_kaggle
b1_kaggle  = T{:, 4};
p2_k1      = T{:, 5};

%% ─── Separate onset and wakeup ───────────────────────────────────────
if iscell(event_col)
    is_onset = strcmp(event_col, 'onset');
else
    is_onset = event_col == categorical("onset");
end

tol_onset     = tol_col(is_onset);
b1_onset      = b1_kaggle(is_onset);
p2_onset      = p2_k1(is_onset);

tol_wakeup    = tol_col(~is_onset);
b1_wakeup     = b1_kaggle(~is_onset);
p2_wakeup     = p2_k1(~is_onset);

% Convert tolerance from steps (5s each) to minutes
tol_onset_min  = tol_onset * 5 / 60;
tol_wakeup_min = tol_wakeup * 5 / 60;

%% ─── IEEE-quality figure ─────────────────────────────────────────────
% Color palette
c_b1 = [0.200, 0.400, 0.800];   % blue
c_p2 = [0.850, 0.325, 0.098];   % vermilion

fig = figure('Units','centimeters','Position',[2 2 18 7]);

% ── Onset subplot ──
ax1 = subplot(1,2,1);
hold on; box on; grid on;
plot(tol_onset_min, b1_onset, '-o', 'Color', c_b1, 'LineWidth', 1.5, ...
    'MarkerSize', 5, 'MarkerFaceColor', c_b1, 'DisplayName', 'B1 (kaggle)');
plot(tol_onset_min, p2_onset, '-s', 'Color', c_p2, 'LineWidth', 1.5, ...
    'MarkerSize', 5, 'MarkerFaceColor', c_p2, 'DisplayName', 'P2 (K=1)');
xlabel('Tolerance (min)', 'FontSize', 9);
ylabel('Average Precision', 'FontSize', 9);
title('(a) Onset', 'FontSize', 10);
legend('Location', 'southeast', 'FontSize', 7);
set(ax1, 'FontSize', 8, 'TickDir', 'out');
xlim([0 max(tol_onset_min)+1]);

% ── Wakeup subplot ──
ax2 = subplot(1,2,2);
hold on; box on; grid on;
plot(tol_wakeup_min, b1_wakeup, '-o', 'Color', c_b1, 'LineWidth', 1.5, ...
    'MarkerSize', 5, 'MarkerFaceColor', c_b1, 'DisplayName', 'B1 (kaggle)');
plot(tol_wakeup_min, p2_wakeup, '-s', 'Color', c_p2, 'LineWidth', 1.5, ...
    'MarkerSize', 5, 'MarkerFaceColor', c_p2, 'DisplayName', 'P2 (K=1)');
xlabel('Tolerance (min)', 'FontSize', 9);
ylabel('Average Precision', 'FontSize', 9);
title('(b) Wakeup', 'FontSize', 10);
legend('Location', 'southeast', 'FontSize', 7);
set(ax2, 'FontSize', 8, 'TickDir', 'out');
xlim([0 max(tol_wakeup_min)+1]);

%% ─── Export ──────────────────────────────────────────────────────────
set(fig, 'PaperPositionMode', 'auto');
print(fig, fullfile(base_dir, 'outputs', 'fig3_per_tolerance_ap'), '-dpdf', '-r600');
print(fig, fullfile(base_dir, 'outputs', 'fig3_per_tolerance_ap'), '-dpng', '-r300');
fprintf('Saved fig3_per_tolerance_ap.pdf/.png\n');
