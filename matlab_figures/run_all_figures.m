% ----------------------------------------------------------------------------
% Structured Semi-Markov Decoding with Conformal Prediction for
% Wrist-Actigraphy Sleep Event Detection
%
% Author    : Hoang Nhut Huynh
% Copyright : (c) 2026 Hoang Nhut Huynh
% ----------------------------------------------------------------------------
%% Master script: generate all IEEE JBHI figures
% Run this from MATLAB with matlab_figures/ as the current folder.
% Each fig script reads CSVs from outputs/ and exports PDF+PNG.

clear; close all; clc;

%% ─── Set your base directory ─────────────────────────────────────────
% Each script resolves base_dir to the repository root automatically and reads
% the CSV files written by the Python pipeline under outputs/.

fprintf('=== Generating IEEE JBHI Figures ===\n\n');

fprintf('[1/4] Fig 3: Per-tolerance AP ...\n');
run('fig3_per_tolerance_ap.m');
fprintf('     Done.\n\n');

fprintf('[2/4] Fig 4: Risk-coverage curve ...\n');
run('fig4_risk_coverage.m');
fprintf('     Done.\n\n');

fprintf('[3/5] Fig 5: Null-state ablation ...\n');
run('fig5_ablation_null.m');
fprintf('     Done.\n\n');

fprintf('[4/5] Fig 6: Conformal coverage ...\n');
run('fig6_conformal_coverage.m');
fprintf('     Done.\n\n');

fprintf('[5/5] Fig 7: Bland-Altman (Newcastle) ...\n');
run('fig7_bland_altman_newcastle.m');
fprintf('     Done.\n\n');

fprintf('=== All figures exported to outputs/ ===\n');
