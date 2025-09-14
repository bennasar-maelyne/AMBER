function [rinit,outputparams] = create_cell_distribution(cells,Nspheres)

    % Create random spheres with stacked, lognormal cell populations
    % Inputs:
    % Ouputs:
    % Josh Marchant, 2025

    % Set rng seed for repeatbility
    rng(4); % default algorithm is twister
    disp('Rng seed is set. Be sure to train with multiple seeds for generalizability!');
    
    % Define cell population parameters
    target_rad = cells.target_rad;
    std_rad = cells.std_rad;
    population_perc = cells.population_perc;
    doplot = cells.doplot;
    %disp(doplot)

    Npop = 2;
    Neachpop = round(Nspheres * population_perc);
    if sum(Neachpop)~= Nspheres
        [largerval,largerindx] = max(Neachpop);
        Neachpop(largerindx) = Nspheres - sum(Neachpop(Neachpop~=largerval));
    end
    outputparams.Neachpop = Neachpop;

    rinit_unshuffled = []; 
    outputparams.mu_log = [];
    outputparams.sigma_log= [];

    for ii = 1:Npop
        
        meanrad_in = target_rad(ii);
        stdrad_in = std_rad(ii);

        var_in = stdrad_in^2;
        mu_log = log(meanrad_in^2 / sqrt(var_in + meanrad_in^2));
        sigma_log = sqrt(log(var_in / meanrad_in^2 + 1));
        outputparams.mu_log = cat(2,outputparams.mu_log,mu_log);
        outputparams.sigma_log = cat(2,outputparams.sigma_log,sigma_log);
        radii = lognrnd(mu_log, sigma_log, [Neachpop(ii), 1]);
        rinit_unshuffled = cat(1,rinit_unshuffled,radii);

        fprintf('Population %d: %d spheres.\n', ii, Neachpop(ii));
    
        fprintf('Target Mean Radius: %.2f,  Generated Mean Radius: %.2f\n', meanrad_in, mean(radii));
        fprintf('Target Std Dev:     %.2f,  Generated Std Dev:     %.2f\n', stdrad_in, std(radii));

    end

    rinit = rinit_unshuffled(randperm(Nspheres));
    
    if doplot
        %disp(doplot)
        figure;
        histogram(rinit, 50, 'Normalization', 'pdf');
        title('Simulated Glioblastoma Cell Radius Distribution (Lognormal)');
        xlabel('Cell Radius ($\mu$m)', 'Interpreter', 'latex');
        ylabel('Probability Density', 'Interpreter', 'latex');
        grid on;
        box on;
        %display(gcf);
    end

end