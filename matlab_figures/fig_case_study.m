% ----------------------------------------------------------------------------
% Structured Semi-Markov Decoding with Conformal Prediction for
% Wrist-Actigraphy Sleep Event Detection
%
% Author    : Hoang Nhut Huynh
% Copyright : (c) 2026 Hoang Nhut Huynh
% ----------------------------------------------------------------------------
%% Fig (case study): qualitative example of structured decoding
% (a) normal night  : anglez signal + onset/wakeup posterior marginals,
%                     MAP pair, 90% HPD intervals, ground truth
% (b) non-wear night: flat signal, p_null > 0.5 -> decoder abstains
%
% Reads : outputs/fig_case_study/{normal,nonwear}.csv + *_meta.csv
%         (produced by: python -m scripts.export_fig_case_study)
% Output: fig_case_study.pdf / .png   (IEEE one-column, 8.9 cm wide)

clear; close all; clc;

%% ─── Config ───────────────────────────────────────────────────────────
base_dir = fileparts(fileparts(mfilename('fullpath')));  % repository root
dat_dir  = fullfile(base_dir, 'outputs', 'fig_case_study');

c_sig = [0.55 0.55 0.55];        % anglez: gray
c_on  = [0.850 0.325 0.098];     % onset : vermilion
c_wk  = [0.200 0.400 0.800];     % wakeup: blue
c_gt  = [0 0 0];                 % ground truth: black dashed

fs = 8;  % font size (pt)

%% ─── Read data ────────────────────────────────────────────────────────
Tn = readtable(fullfile(dat_dir, 'normal.csv'));
Mn = readtable(fullfile(dat_dir, 'normal_meta.csv'));
Tw = readtable(fullfile(dat_dir, 'nonwear.csv'));
Mw = readtable(fullfile(dat_dir, 'nonwear_meta.csv'));

%% ─── Figure: one column, two stacked panels ───────────────────────────
fig = figure('Units','centimeters','Position',[2 2 8.9 9.5]);
tl = tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

% ===== (a) normal night ================================================
ax1 = nexttile; hold(ax1,'on');
[hh, ymaxA] = plot_panel(ax1, Tn, Mn, c_sig, c_on, c_wk, c_gt, fs, true, []);
title(ax1, sprintf('(a) Normal night (p_%c = %.2f)', ...
      char(8709), Mn.p_null), 'FontSize', fs, 'FontWeight','normal');

% ===== (b) non-wear night ==============================================
ax2 = nexttile; hold(ax2,'on');
plot_panel(ax2, Tw, Mw, c_sig, c_on, c_wk, c_gt, fs, false, ymaxA);
title(ax2, sprintf('(b) Non-wear night (p_%c = %.2f %c abstain)', ...
      char(8709), Mw.p_null, char(8594)), ...
      'FontSize', fs, 'FontWeight','normal');
xlabel(ax2, 'Time within window (h)', 'FontSize', fs);

% shared legend (handles returned by plot_panel for panel a)
lg = legend(ax1, hh, {'angle-z', 'onset posterior', 'wakeup posterior', ...
     '90% HPD', 'MAP', 'ground truth'}, ...
     'FontSize', fs-1, 'NumColumns', 3, 'Location','northoutside');
lg.Box = 'off';

%% ─── Export ───────────────────────────────────────────────────────────
set(findall(fig,'-property','FontName'), 'FontName','Times New Roman');
exportgraphics(fig, fullfile(base_dir,'outputs','fig_case_study.pdf'), ...
    'ContentType','vector');
exportgraphics(fig, fullfile(base_dir,'outputs','fig_case_study.png'), ...
    'Resolution', 600);
fprintf('saved fig_case_study.pdf / .png\n');

%% ─── Panel helper ─────────────────────────────────────────────────────
function [hh, ymax] = plot_panel(ax, T, M, c_sig, c_on, c_wk, c_gt, fs, show_marks, ymax_in)
    t = T.t_hr;

    % left axis: angle-z signal
    yyaxis(ax,'left');
    h_sig = plot(ax, t, T.anglez, '-', 'Color', c_sig, 'LineWidth', 0.5);
    ylabel(ax, 'angle-z (deg)', 'FontSize', fs);
    ax.YColor = c_sig * 0.6;
    ylim(ax, [-95 95]);

    % right axis: posterior marginals
    yyaxis(ax,'right');
    if isempty(ymax_in)
        ymax = max([T.m_on; T.m_wk; 1e-3]) * 1.15;
    else
        ymax = ymax_in;   % reuse panel (a) scale so (b) reads as 'zero mass'
    end

    if show_marks
        % 90% HPD bands (behind the curves)
        [bx, by] = hband(M.hpd_on_lo_hr, M.hpd_on_hi_hr, ymax);
        h_hpd = patch(ax, bx, by, c_on, ...
              'FaceAlpha', 0.12, 'EdgeColor','none');
        [bx, by] = hband(M.hpd_wk_lo_hr, M.hpd_wk_hi_hr, ymax);
        patch(ax, bx, by, c_wk, 'FaceAlpha', 0.12, 'EdgeColor','none');
    else
        h_hpd = patch(ax, nan(1,4), nan(1,4), c_on, 'EdgeColor','none');
    end

    % posterior curves (filled)
    h_on = fill(ax, [t; flipud(t)], [T.m_on; zeros(size(t))], c_on, ...
         'FaceAlpha', 0.45, 'EdgeColor', c_on, 'LineWidth', 0.7, ...
         'LineStyle','-');
    h_wk = fill(ax, [t; flipud(t)], [T.m_wk; zeros(size(t))], c_wk, ...
         'FaceAlpha', 0.45, 'EdgeColor', c_wk, 'LineWidth', 0.7, ...
         'LineStyle','-');

    if show_marks
        h_map = xline(ax, M.map_on_hr, '-',  'Color', c_on, 'LineWidth', 1.0);
        xline(ax, M.map_wk_hr, '-',  'Color', c_wk, 'LineWidth', 1.0);
        h_gt = xline(ax, M.gt_on_hr,  '--', 'Color', c_gt, 'LineWidth', 0.8);
        if ~isnan(M.gt_wk_hr)
            xline(ax, M.gt_wk_hr, '--', 'Color', c_gt, 'LineWidth', 0.8);
        end
    else
        h_map = gobjects(1); h_gt = gobjects(1);
        text(ax, mean(t), ymax*0.92, 'no event emitted', ...
             'FontSize', fs, 'FontAngle','italic', ...
             'HorizontalAlignment','center', 'Color', [0.3 0.3 0.3]);
    end

    ylabel(ax, 'posterior mass / bin', 'FontSize', fs);
    ax.YColor = [0.2 0.2 0.2];
    ylim(ax, [0 ymax]);
    xlim(ax, [t(1) t(end)]);
    set(ax, 'FontSize', fs, 'TickDir','out', 'Box','off');

    hh = [h_sig h_on h_wk h_hpd h_map h_gt];
end

function [x, y] = hband(lo, hi, ymax)
    % patch coordinates for a vertical band [lo,hi] x [0,ymax]
    x = [lo hi hi lo];
    y = [0 0 ymax ymax];
end
