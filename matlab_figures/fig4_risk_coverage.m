% ----------------------------------------------------------------------------
% Structured Semi-Markov Decoding with Conformal Prediction for
% Wrist-Actigraphy Sleep Event Detection
%
% Author    : Hoang Nhut Huynh
% Copyright : (c) 2026 Hoang Nhut Huynh
% ----------------------------------------------------------------------------
%% Fig 4: Risk–Coverage Curve (Selective Prediction)
% Reads: outputs/p2_main/seed{42,43,44}/p3/risk_coverage.csv
% Output: fig4_risk_coverage.pdf
%
% Plots 3 criteria: interval_width, p_null, peak_mass
% If 3 seeds available: plots mean ± shaded std

clear; close all; clc;

%% ─── Config ───────────────────────────────────────────────────────────
base_dir = fileparts(fileparts(mfilename('fullpath')));  % repository root
seeds = [42, 43, 44];
criteria_order = {'interval width', 'p_null', 'peak mass'};
criteria_labels = {'Interval width', 'p_{null}', 'Peak mass'};
criteria_colors = {[0.850, 0.325, 0.098], ...   % vermilion
                   [0.200, 0.400, 0.800], ...   % blue
                   [0.467, 0.675, 0.188]};      % green

%% ─── Read all seeds ──────────────────────────────────────────────────
all_data = struct();
for ci = 1:numel(criteria_order)
    cname = criteria_order{ci};
    all_data(ci).name = cname;
    all_data(ci).coverage = [];
    all_data(ci).risk = [];
end

n_seeds_found = 0;
for si = 1:numel(seeds)
    csv_path = fullfile(base_dir, 'outputs', 'p2_main', ...
        sprintf('seed%d', seeds(si)), 'p3', 'risk_coverage.csv');
    if ~isfile(csv_path)
        fprintf('Skipping seed %d (file not found)\n', seeds(si));
        continue;
    end
    n_seeds_found = n_seeds_found + 1;

    T = readtable(csv_path, 'TextType', 'string');

    for ci = 1:numel(criteria_order)
        cname = criteria_order{ci};
        mask = T.criterion == cname;
        cov_frac = T.coverage_frac(mask);
        risk     = T.pct_err_gt30(mask);

        if n_seeds_found == 1
            all_data(ci).coverage = cov_frac(:)';
            all_data(ci).risk     = risk(:)';
        else
            % Align by coverage_frac (should match across seeds)
            all_data(ci).risk = [all_data(ci).risk; risk(:)'];
        end
    end
end

fprintf('Found %d seeds\n', n_seeds_found);

%% ─── Plot ────────────────────────────────────────────────────────────
fig = figure('Units','centimeters','Position',[2 2 9 7.5]);
ax = axes; hold on; box on; grid on;

for ci = 1:numel(criteria_order)
    x = all_data(ci).coverage * 100;  % to percentage
    y_all = all_data(ci).risk;

    if size(y_all, 1) > 1
        y_mean = mean(y_all, 1);
        y_std  = std(y_all, 0, 1);
        % Shaded region
        fill([x, fliplr(x)], ...
             [y_mean - y_std, fliplr(y_mean + y_std)], ...
             criteria_colors{ci}, 'FaceAlpha', 0.15, ...
             'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(x, y_mean, '-', 'Color', criteria_colors{ci}, ...
            'LineWidth', 1.8, 'DisplayName', criteria_labels{ci});
    else
        plot(x, y_all, '-', 'Color', criteria_colors{ci}, ...
            'LineWidth', 1.8, 'DisplayName', criteria_labels{ci});
    end
end

% Reference: full-set error rate (coverage = 100%)
% Draw as horizontal dashed line at rightmost risk value
for ci = 1:numel(criteria_order)
    y_all = all_data(ci).risk;
    if ~isempty(y_all)
        full_risk = y_all(1, end);  % last coverage point ≈ 100%
        if ci == 1  % only draw once
            yline(full_risk, '--', 'Color', [0.5 0.5 0.5], ...
                'LineWidth', 1, 'DisplayName', 'Full set', ...
                'Alpha', 0.7);
        end
        break;
    end
end

xlabel('Retention (%)', 'FontSize', 9);
ylabel('Nights with >30 min error (%)', 'FontSize', 9);
legend('Location', 'northwest', 'FontSize', 7);
set(ax, 'FontSize', 8, 'TickDir', 'out');
xlim([45 105]);

%% ─── Export ──────────────────────────────────────────────────────────
set(fig, 'PaperPositionMode', 'auto');
print(fig, fullfile(base_dir, 'outputs', 'fig4_risk_coverage'), '-dpdf', '-r600');
print(fig, fullfile(base_dir, 'outputs', 'fig4_risk_coverage'), '-dpng', '-r300');
fprintf('Saved fig4_risk_coverage.pdf/.png\n');
