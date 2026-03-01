%% Extraction of the Lookup Table - Octave Compatible

% Configuration
root = 'C:/Users/maely/OneDrive/Bureau/ENSTA/CESURE/AMBER/monte-carlo-simulation-sphere-PGSE-main/data';
projname = 'AMBER_maelyne';
population_type = 'lognorm';
population_label = [population_type, '_population_RealisticTests2_V3'];
projectpath = fullfile(root, projname, population_label);
pkg load statistics

lut_file = fullfile(projectpath, 'lookup_table_validation_compat.mat');

% If the lookup table already exists
if exist(lut_file,'file')
    fprintf('Loading existing lookup table...\n');
    load(lut_file, 'params', 'signals_4D', 'sequence');
else
    fprintf('Lookup table not found. Running generation...\n');
    params_file = fullfile(projectpath, 'params_in.txt');

    if exist(params_file, 'file')
        % Octave : remplacer readmatrix par dlmread
        params_data = dlmread(params_file);
        Ncombos = size(params_data, 1);
    else
        error('params_in.txt not found in %s\n', projectpath);
    end

    simdir = fullfile(projectpath, 'sphere_0001'); % assuming b-values/directions same for all

    % Load bvals and bvecs
    btab_raw = dlmread(fullfile(simdir, 'gradient_bvalstab.txt'));
    [C, IA, IC] = unique(btab_raw);
    bval = C / 1000;
    n_bvec = IA(2) - IA(1);
    Btab = dlmread(fullfile(simdir, 'gradient_btab.txt'));
    btab = reshape(Btab, 4, []);  % reshape as 4×(n_bval*n_bvec)
    bvecs = btab(2:end,1:n_bvec).';

    % Load diffusion times and pulse widths
    TD = dlmread(fullfile(simdir, 'TD_array.txt'));
    pulse_widths = dlmread(fullfile(simdir, 'gradient_DELdel.txt'));
    pulse_widths = reshape(pulse_widths, 2, []).';
    delta = pulse_widths(:,2);

    % Derived values
    n_bval = numel(bval);
    n_del = numel(TD);
    n_signals = n_bvec * n_bval * n_del;

    % Sequence struct
    sequence.bval = bval;
    sequence.bvecs = bvecs;
    sequence.n_bval = n_bval;
    sequence.n_bvec = n_bvec;
    sequence.n_del = n_del;
    sequence.TD = TD;
    sequence.delta = delta(1:n_del);

    % Output arrays
    params = zeros(Ncombos, 4);          % f, Dex, rmean, rsd
    signals_4D = zeros(Ncombos, n_del, n_bval, n_bvec);  % [i × d × b × v]

    % Loop over simulations
    for i = 1:Ncombos
        simname = sprintf('sphere_%04d', i);
        simdir = fullfile(projectpath, simname);
        if ~exist(simdir, 'dir')
            warning(['Simulation folder ', simdir, ' does not exist. Skipping simulation ', num2str(i)]);
            continue;
        end

        % --- Load input parameters ---
        parafile = fullfile(simdir, 'sim_para.txt');
        if ~exist(parafile,'file')
            warning(['Missing sim_para.txt for simulation ', num2str(i)]);
            continue;
        end
        paravalues = dlmread(parafile);
        Din = paravalues(4);
        Dex = paravalues(5);

        % --- Load population parameters ---
        pop_para_file = fullfile(simdir, 'population_para.txt');
        if ~exist(pop_para_file,'file')
            warning(['Missing population_para.txt for simulation ', num2str(i)]);
            continue;
        end
        pop_para = dlmread(pop_para_file);
        rmean = pop_para(1);
        rsd = pop_para(2);
        pop_perc = pop_para(3);
        f = pop_para(4);

        % --- Store parameters ---
        params(i,:) = [f, Dex, rmean, rsd];

        % --- Load signals ---
        sigfile = fullfile(simdir, 'sig_diffusion.txt');
        if ~exist(sigfile,'file')
            warning(['Missing sig_diffusion.txt for simulation ', num2str(i)]);
            continue;
        end
        sig = dlmread(sigfile);

        if numel(sig) ~= n_signals
            warning(['Unexpected signal length for simulation ', num2str(i)]);
            continue;
        end

        sig(sig < 1e-6) = 1e-3;  % avoid negative signals

        % Reshape to 3D [nDel × nBval × nBvec]
        sig3D = reshape(sig, [n_bvec, n_bval, n_del]);
        sig3D = permute(sig3D, [3 2 1]);

        % Store in 4D array
        signals_4D(i,:,:,:) = sig3D;
    end

    % Save lookup table
    save('-v7', 'lookup_table_validation_compat.mat', 'params', 'signals_4D', 'sequence');
    fprintf('Lookup table saved successfully.\n');
    save(lut_file, 'params', 'signals_4D', 'sequence');
end

%% Check the Lookup Table values
%disp('--- b values ---'); disp(sequence.bval);
%disp('--- Diffusion times ---'); disp(sequence.TD);
%disp('--- Gradient directions ---'); disp(sequence.bvecs);

disp('--- First rows of params [f, Dex, rmean, rsd] ---');
disp(params(1:min(20,end), :));

disp('--- Sequence structure ---');

disp('b-values:');
disp(sequence.bval(1:min(5,end)));

disp('Diffusion times TD:');
disp(sequence.TD(1:min(5,end)));

disp('Gradient directions (first 5):');
disp(sequence.bvecs(1:min(5,end), :));

disp('--- Size of signals_4D ---');
disp(size(signals_4D));


%% Multivariate analysis (Octave compatible)

% -----------------------------
% Data preparation
% -----------------------------
[n_sim, nDel, nBval, nBvec] = size(signals_4D);

X = [];  % Features: f, Dex, rmean, rsd, bval, TD
y = [];  % Signal (direction-averaged)

for td_idx = 1:nDel
    for b_idx = 1:nBval
        sig_avg = squeeze(mean(signals_4D(:, td_idx, b_idx, :), 4));  % [n_sim × 1]

        X_temp = [params(:,1:4), ...
                  repmat(sequence.bval(b_idx), n_sim, 1), ...
                  repmat(sequence.TD(td_idx), n_sim, 1)];

        X = [X; X_temp];
        y = [y; sig_avg];
    end
end

param_names = {'f', 'Dex', 'rmean', 'rsd', 'bval', 'TD'};

% -----------------------------
% Min–max normalization
% -----------------------------
y_min = min(y);
y_max = max(y);
y_norm = (y - y_min) ./ (y_max - y_min);

% -----------------------------
% Correlation matrix
% -----------------------------
data_mat = [X y_norm];
corr_matrix = corrcoef(data_mat);

disp('Correlation Matrix:');
disp(corr_matrix);

% -----------------------------
% Heatmap
% -----------------------------
figure;
imagesc(corr_matrix);
colormap(jet);
colorbar;
title('Correlation Matrix');

n_labels = numel(param_names) + 1;
xticks(1:n_labels);
yticks(1:n_labels);
xticklabels([param_names, {'Signal'}]);
yticklabels([param_names, {'Signal'}]);
set(gca, 'FontSize', 12, 'XTickLabelRotation', 45);

% Display numeric values
for i = 1:n_labels
    for j = 1:n_labels
        text(j, i, sprintf('%.2f', corr_matrix(i,j)), ...
            'HorizontalAlignment', 'center', 'Color', 'k');
    end
end

if ~exist('Figures','dir'), mkdir('Figures'); end
saveas(gcf, 'Figures/Corrmatrix.png');

% -----------------------------
% Boxplots (discretized)
% -----------------------------
figure;
bval_groups = round(X(:,5)*100)/100;  % discretize b-values
boxplot(y_norm, bval_groups);
xlabel('b-value (ms/µm²)');
ylabel('Normalized signal');
title('Signal vs b-value');
saveas(gcf, 'Figures/Boxplot_signal_bval.png');

figure;
TD_groups = X(:,6);
boxplot(y_norm, TD_groups);
xlabel('Diffusion Time (ms)');
ylabel('Normalized signal');
title('Signal vs Diffusion Time');
saveas(gcf, 'Figures/Boxplot_signal_TD.png');

% -----------------------------
% Histograms (Octave-style)
% -----------------------------
figure;
subplot(2,3,1); hist(X(:,1), 20); title('f (cell density)');
subplot(2,3,2); hist(X(:,2), 20); title('Dex');
subplot(2,3,3); hist(X(:,3), 20); title('rmean');
subplot(2,3,4); hist(X(:,4), 20); title('rsd');
subplot(2,3,5); hist(X(:,5), 20); title('b-value');
subplot(2,3,6); hist(y, 20);       title('Signal Intensity');

