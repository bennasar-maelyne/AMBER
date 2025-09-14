function [rinit, kappa_init, cellType, outputparams] = create_cell_distribution_V2(cells, Nspheres, f_alive, kappa_alive, kappa_dead)

    rng(4); 
    
    % Parameters
    target_rad = cells.target_rad;
    std_rad = cells.std_rad;
    population_perc = cells.population_perc; % for multiple populations
    doplot = cells.doplot;

    Npop = length(target_rad);
    Neachpop = round(Nspheres * population_perc);
    if sum(Neachpop)~= Nspheres
        [largerval,largerindx] = max(Neachpop);
        Neachpop(largerindx) = Nspheres - sum(Neachpop(Neachpop~=largerval));
    end
    outputparams.Neachpop = Neachpop;

    rinit_unshuffled = [];
    outputparams.mu_log = [];
    outputparams.sigma_log = [];

    % Lognormal radius distribution
    for ii = 1:Npop
        meanrad_in = target_rad(ii);
        stdrad_in = std_rad(ii);

        var_in = stdrad_in^2;
        mu_log = log(meanrad_in^2 / sqrt(var_in + meanrad_in^2));
        sigma_log = sqrt(log(var_in / meanrad_in^2 + 1));
        outputparams.mu_log = [outputparams.mu_log, mu_log];
        outputparams.sigma_log = [outputparams.sigma_log, sigma_log];

        radii = lognrnd(mu_log, sigma_log, [Neachpop(ii), 1]);
        rinit_unshuffled = [rinit_unshuffled; radii];
    end

    % Shuffle
    rinit = rinit_unshuffled(randperm(Nspheres));

    % Cell type attribution
    cellType = zeros(Nspheres,1);
    n_dead = round(Nspheres * (1 - f_alive));
    dead_idx = randperm(Nspheres, n_dead);
    cellType(dead_idx) = 1;

    % Swelling for dead cells
    swelling_factor = 1.2;
    rinit(cellType==1) = rinit(cellType==1) * swelling_factor;

    % Permeability per cell
    kappa_init = kappa_alive * ones(Nspheres,1);
    kappa_init(cellType==1) = kappa_dead;

end
