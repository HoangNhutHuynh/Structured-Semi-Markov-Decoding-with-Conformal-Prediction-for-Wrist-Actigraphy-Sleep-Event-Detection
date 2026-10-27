% ----------------------------------------------------------------------------
% Structured Semi-Markov Decoding with Conformal Prediction for
% Wrist-Actigraphy Sleep Event Detection
%
% Author    : Hoang Nhut Huynh
% Copyright : (c) 2026 Hoang Nhut Huynh
% ----------------------------------------------------------------------------
%% Fig 5: Conformal Coverage vs Nominal Level
% Reads: outputs/p2_main/seed{42,43,44}/p3/conformal_coverage.csv
%        outputs/newcastle/results/conformal_transfer.csv
%        outputs/newcastle/results/conformal_loso.csv
% Output: fig5_conformal_coverage.pdf
%
% Shows: CMI cross-conformal (abs & HPD), Newcastle transfer, Newcastle LOSO

clear; close all; clc;

%% ─── Config ───────────────────────────────────────────────────────────
base_dir = fileparts(fileparts(mfilename('fullpath')));  % repository root
seeds = [42, 43, 44];

%% ─── Read CMI conformal coverage (cross-conformal, all seeds) ─────────
cmi_abs = struct('target',[],'coverage',[],'width_on',[],'width_wk',[]);
cmi_hpd = struct('target',[],'coverage',[],'width_on',[],'width_wk',[]);

for si = 1:numel(seeds)
    csv_path = fullfile(base_dir, 'outputs', 'p2_main', ...
        sprintf('seed%d', seeds(si)), 'p3', 'conformal_coverage.csv');
    if ~isfile(csv_path), continue; end

    T = readtable(csv_path, 'TextType', 'string');

    % Filter cross-conformal only
    cross = T(T.calibration == "cross", :);

    % ABS onset
    abs_on = cross(cross.score == "abs" & cross.event == "onset", :);
    hpd_on = cross(cross.score == "hpd" & cross.event == "onset", :);
    abs_wk = cross(cross.score == "abs" & cross.event == "wakeup", :);
    hpd_wk = cross(cross.score == "hpd" & cross.event == "wakeup", :);

    if si == 1
        cmi_abs.target = abs_on.target;
        cmi_abs.cov_on = abs_on.coverage;
        cmi_abs.cov_wk = abs_wk.coverage;
        cmi_abs.width_on = abs_on.median_width_min;
        cmi_abs.width_wk = abs_wk.median_width_min;

        cmi_hpd.target = hpd_on.target;
        cmi_hpd.cov_on = hpd_on.coverage;
        cmi_hpd.cov_wk = hpd_wk.coverage;
        cmi_hpd.width_on = hpd_on.median_width_min;
        cmi_hpd.width_wk = hpd_wk.median_width_min;
    else
        cmi_abs.cov_on = [cmi_abs.cov_on, abs_on.coverage];
        cmi_abs.cov_wk = [cmi_abs.cov_wk, abs_wk.coverage];
        cmi_hpd.cov_on = [cmi_hpd.cov_on, hpd_on.coverage];
        cmi_hpd.cov_wk = [cmi_hpd.cov_wk, hpd_wk.coverage];
    end
end

%% ─── Read Newcastle conformal transfer & LOSO ───────────────────────
nc_transfer_path = fullfile(base_dir, 'outputs', 'newcastle', 'results', 'conformal_transfer.csv');
nc_loso_path     = fullfile(base_dir, 'outputs', 'newcastle', 'results', 'conformal_loso.csv');

nc_transfer = [];
nc_loso = [];
if isfile(nc_transfer_path)
    nc_transfer = readtable(nc_transfer_path, 'TextType', 'string');
end
if isfile(nc_loso_path)
    nc_loso = readtable(nc_loso_path, 'TextType', 'string');
end

%% ─── Figure: 2-panel (onset, wakeup) ────────────────────────────────
c_abs  = [0.200, 0.400, 0.800];  % blue
c_hpd  = [0.850, 0.325, 0.098];  % vermilion
c_nc_t = [0.467, 0.675, 0.188];  % green
c_nc_l = [0.580, 0.000, 0.827];  % purple

fig = figure('Units','centimeters','Position',[2 2 18 7.5]);

for panel = 1:2
    ax = subplot(1,2,panel); hold on; box on; grid on;

    event = {'onset','wakeup'};
    ev = event{panel};

    % ── CMI cross-conformal ──
    if panel == 1
        targets_abs = cmi_abs.target;
        cov_abs = mean(cmi_abs.cov_on, 2);
        cov_hpd = mean(cmi_hpd.cov_on, 2);
    else
        targets_abs = cmi_abs.target;
        cov_abs = mean(cmi_abs.cov_wk, 2);
        cov_hpd = mean(cmi_hpd.cov_wk, 2);
    end

    plot(targets_abs, cov_abs, 'o', 'Color', c_abs, 'MarkerSize', 8, ...
        'LineWidth', 1.5, 'MarkerFaceColor', c_abs, ...
        'DisplayName', 'CMI absolute');
    plot(targets_abs, cov_hpd, 's', 'Color', c_hpd, 'MarkerSize', 8, ...
        'LineWidth', 1.5, 'MarkerFaceColor', c_hpd, ...
        'DisplayName', 'CMI HPD');

    % ── Newcastle transfer ──
    if ~isempty(nc_transfer)
        nc_t = nc_transfer(nc_transfer.event == ev, :);
        plot(nc_t.target, nc_t.newcastle_coverage, '^', ...
            'Color', c_nc_t, 'MarkerSize', 8, 'LineWidth', 1.5, ...
            'MarkerFaceColor', c_nc_t, ...
            'DisplayName', 'Newcastle transfer');
    end

    % ── Newcastle LOSO ──
    if ~isempty(nc_loso)
        nc_l = nc_loso(nc_loso.event == ev, :);
        plot(nc_l.target, nc_l.loso_coverage, 'd', ...
            'Color', c_nc_l, 'MarkerSize', 8, 'LineWidth', 1.5, ...
            'MarkerFaceColor', c_nc_l, ...
            'DisplayName', 'Newcastle LOSO');
    end

    % ── Ideal line ──
    plot([0.85 1], [0.85 1], '--', 'Color', [0.5 0.5 0.5], ...
        'LineWidth', 1, 'HandleVisibility', 'off');

    xlabel('Nominal coverage (1-\alpha)', 'FontSize', 9);
    ylabel('Empirical coverage', 'FontSize', 9);
    if panel == 1
        title('(a) Onset', 'FontSize', 10);
    else
        title('(b) Wakeup', 'FontSize', 10);
    end
    legend('Location', 'southeast', 'FontSize', 6.5);
    set(ax, 'FontSize', 8, 'TickDir', 'out');
    xlim([0.84 1.01]); ylim([0.2 1.05]);
end

%% ─── Export ──────────────────────────────────────────────────────────
set(fig, 'PaperPositionMode', 'auto');
print(fig, fullfile(base_dir, 'outputs', 'fig5_conformal_coverage'), '-dpdf', '-r600');
print(fig, fullfile(base_dir, 'outputs', 'fig5_conformal_coverage'), '-dpng', '-r300');
fprintf('Saved fig5_conformal_coverage.pdf/.png\n');
