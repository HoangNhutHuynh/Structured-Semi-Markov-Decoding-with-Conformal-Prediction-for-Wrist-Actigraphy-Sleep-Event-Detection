% ----------------------------------------------------------------------------
% Structured Semi-Markov Decoding with Conformal Prediction for
% Wrist-Actigraphy Sleep Event Detection
%
% Author    : Hoang Nhut Huynh
% Copyright : (c) 2026 Hoang Nhut Huynh
% ----------------------------------------------------------------------------
%% Fig 5: Null-State Ablation — Impact on Detection and Clinical Metrics
% Compares P2 (with null) vs P2 (no null) across key metrics.
% Data: both from 3-seed mean +/- SD (seeds 42, 43, 44).
% Output: fig5_ablation_null.pdf/.png

clear; close all; clc;

%% --- Config -----------------------------------------------------------
base_dir = fileparts(fileparts(mfilename('fullpath')));  % repository root

%% --- Data -------------------------------------------------------------
% P2 with null (3-seed mean +/- sd, from seed_summary.csv)
p2_edap      = 0.721;   p2_edap_sd   = 0.003;
p2_k1        = 0.669;   p2_k1_sd     = 0.003;
p2_fd        = 0.000;   p2_fd_sd     = 0.000;
p2_bias      = 3.46;    p2_bias_sd   = 2.24;
p2_loa_lo    = -102.52; p2_loa_hi    = 109.45;
p2_loa_width = p2_loa_hi - p2_loa_lo;    % ~212 min

% P2 no null (3-seed mean +/- sd, from metrics.json seeds 42,43,44)
nn_edap      = 0.651;   nn_edap_sd   = 0.012;
nn_k1        = 0.603;   nn_k1_sd     = 0.013;
nn_fd        = 1.000;   nn_fd_sd     = 0.000;
nn_bias      = 4.36;    nn_bias_sd   = 1.25;
nn_loa_lo    = -106.90; nn_loa_hi    = 115.63;
nn_loa_width = nn_loa_hi - nn_loa_lo;    % ~222.5 min

%% --- Color palette ----------------------------------------------------
c_null    = [0.200, 0.400, 0.800];   % blue  -- with null
c_nonull  = [0.850, 0.325, 0.098];   % vermilion -- no null

%% --- Figure: 1x3 subplots --------------------------------------------
fig = figure('Units','centimeters','Position',[2 2 18 6.5]);

% -- (a) Detection metrics: EDAP and K1 --
ax1 = subplot(1,3,1);
vals = [p2_edap, nn_edap; p2_k1, nn_k1];
errs = [p2_edap_sd, nn_edap_sd; ...
        p2_k1_sd,   nn_k1_sd];
b = bar(vals, 'grouped');
b(1).FaceColor = c_null;
b(2).FaceColor = c_nonull;
hold on;
% Error bars
ngroups = size(vals, 1);
nbars = size(vals, 2);
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
end
errorbar(x', vals, errs, '.k', 'LineWidth', 0.8, 'CapSize', 4);
set(ax1, 'XTickLabel', {'EDAP', 'K1'}, 'FontSize', 8, 'TickDir', 'out');
ylabel('Score', 'FontSize', 9);
title('(a) Detection', 'FontSize', 10);
ylim([0.5 0.8]);
box on;

% -- (b) Proxy false-detection rate --
ax2 = subplot(1,3,2);
vals_fd = [p2_fd; nn_fd];
b2 = bar(vals_fd, 0.5);
b2.FaceColor = 'flat';
b2.CData(1,:) = c_null;
b2.CData(2,:) = c_nonull;
set(ax2, 'XTickLabel', {'With null', 'No null'}, 'FontSize', 8, 'TickDir', 'out');
ylabel('FD rate', 'FontSize', 9);
title('(b) False detection', 'FontSize', 10);
ylim([0 1.15]);
box on;
% Add value labels on bars
text(1, p2_fd + 0.05, sprintf('%.3f', p2_fd), 'HorizontalAlignment', 'center', ...
    'FontSize', 7, 'FontWeight', 'bold');
text(2, nn_fd + 0.05, sprintf('%.3f', nn_fd), 'HorizontalAlignment', 'center', ...
    'FontSize', 7, 'FontWeight', 'bold');

% -- (c) SPT agreement: bias and LoA width --
ax3 = subplot(1,3,3);
vals_spt = [p2_bias, nn_bias; p2_loa_width, nn_loa_width];
errs_spt = [p2_bias_sd, nn_bias_sd; 0, 0];
b3 = bar(vals_spt, 'grouped');
b3(1).FaceColor = c_null;
b3(2).FaceColor = c_nonull;
hold on;
x3 = nan(2, 2);
for i = 1:2
    x3(i,:) = b3(i).XEndPoints;
end
errorbar(x3', vals_spt, errs_spt, '.k', 'LineWidth', 0.8, 'CapSize', 4);
set(ax3, 'XTickLabel', {'Bias', 'LoA width'}, 'FontSize', 8, 'TickDir', 'out');
ylabel('Minutes', 'FontSize', 9);
title('(c) SPT agreement', 'FontSize', 10);
box on;

% -- Shared legend (top of panel a) --
legend(ax1, {'With null', 'No null'}, 'FontSize', 7, ...
    'Location', 'northeast');

%% --- Export -----------------------------------------------------------
set(fig, 'PaperPositionMode', 'auto');
out_path = fullfile(base_dir, 'outputs', 'fig5_ablation_null');
print(fig, out_path, '-dpdf', '-r600');
print(fig, out_path, '-dpng', '-r300');
fprintf('Saved fig5_ablation_null.pdf/.png\n');
