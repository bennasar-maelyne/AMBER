classdef simul3Dsphere_cuda_pgse_bvec_necrosis < handle
    
    properties (GetAccess = public, SetAccess = public)
        TD;
        dtime; Tstep; NPar; Din; Dex; 
        kappa_alive; kappa_dead; f_alive; % New properties
        pinit;
        dx2t; dx4t; sig;
        % New properties on separated populations
        NParICS_alive; NParICS_dead; NParICS_total;
        Del; del; bval; bvec; NDel; Nbtab;
        % Statistics on populations
        n_alive; n_dead; percent_alive; percent_dead;
    end
    
    properties (GetAccess = private, SetAccess = private)
        D_cnt = [1 2 2 1 2 1];
        W_cnt = [1 4 4 6 12 6 4 12 12 4 1 4 6 4 1];
        D_ind = [1 1; 1 2; 1 3; 2 2; 2 3; 3 3];
        W_ind = [1 1 1 1; 1 1 1 2; 1 1 1 3; 1 1 2 2; 1 1 2 3;
                1 1 3 3; 1 2 2 2; 1 2 2 3; 1 2 3 3; 1 3 3 3;
                2 2 2 2; 2 2 2 3; 2 2 3 3; 2 3 3 3; 3 3 3 3];
    end
    
    methods (Access = public)
        function this = simul3Dsphere_cuda_pgse_bvec_necrosis(root)
            if ~isempty(root)
                this.readJob(root);
            end
        end
        
        function this = readJob(this,root)
            % Load simulation parameters
            param = load(fullfile(root,'sim_para.txt'));
            this.dtime = param(1);
            this.Tstep = param(2);
            this.NPar = param(3);
            this.Din = param(4);
            this.Dex = param(5);
            this.kappa_alive = param(6);  % New parameters
            this.kappa_dead = param(7);   % New parameters
            this.f_alive = param(8);      % New parameters
            this.pinit = param(9);        
            
            % Load time points
            this.TD = load(fullfile(root,'diff_time.txt'));
            
            % Load diffusion tensors 
            this.dx2t = load(fullfile(root,'dx2_diffusion.txt')).*this.D_cnt/this.NPar;
            this.dx4t = load(fullfile(root,'dx4_diffusion.txt')).*this.W_cnt/this.NPar;
            
            % Load population statistics for each time point
            if exist(fullfile(root,'NParICS_alive.txt'), 'file')
                this.NParICS_alive = load(fullfile(root,'NParICS_alive.txt'))/this.NPar;
            else
                warning('File NParICS_alive.txt not found. Setting to zeros.');
                this.NParICS_alive = zeros(size(this.TD));
            end
            
            if exist(fullfile(root,'NParICS_dead.txt'), 'file')
                this.NParICS_dead = load(fullfile(root,'NParICS_dead.txt'))/this.NPar;
            else
                warning('File NParICS_dead.txt not found. Setting to zeros.');
                this.NParICS_dead = zeros(size(this.TD));
            end
            
            % Load total ICS population (should equal alive + dead)
            this.NParICS_total = load(fullfile(root,'NParICS.txt'))/this.NPar;
            
            % Load population statistics from population_stats.txt if available
            if exist(fullfile(root,'population_stats.txt'), 'file')
                this.readPopulationStats(fullfile(root,'population_stats.txt'));
            else
                warning('File population_stats.txt not found. Population stats not loaded.');
            end
            
            % Load gradient parameters
            DELdel = load(fullfile(root,'gradient_DELdel.txt'));
            DELdel = reshape(DELdel,2,[]).';
            this.Del = DELdel(:,1);
            this.del = DELdel(:,2);
            this.NDel = numel(this.Del);

            btab = load(fullfile(root,'gradient_btab.txt'));
            btab = reshape(btab,4,[]).';
            this.bval = btab(:,1);
            this.bvec = btab(:,2:4);
            this.Nbtab = numel(this.bval);

            % Load signal data
            this.sig = load(fullfile(root,'sig_diffusion.txt'))/this.NPar;
            this.sig = reshape(this.sig,this.Nbtab,this.NDel);
        end
        
        function readPopulationStats(this, filename)
            % Read statistics from the text file
            fid = fopen(filename, 'r');
            if fid == -1
                warning('Cannot open population_stats.txt');
                return;
            end
            
            try
                % Extract information line by line 
                while ~feof(fid)
                    line = fgetl(fid);
                    if contains(line, 'Alive cells:')
                        % Number and pourcentage
                        tokens = regexp(line, 'Alive cells: (\d+) \(([0-9.]+)%\)', 'tokens');
                        if ~isempty(tokens)
                            this.n_alive = str2double(tokens{1}{1});
                            this.percent_alive = str2double(tokens{1}{2});
                        end
                    elseif contains(line, 'Dead cells:')
                        tokens = regexp(line, 'Dead cells: (\d+) \(([0-9.]+)%\)', 'tokens');
                        if ~isempty(tokens)
                            this.n_dead = str2double(tokens{1}{1});
                            this.percent_dead = str2double(tokens{1}{2});
                        end
                    end
                end
            catch
                warning('Error reading population_stats.txt');
            end
            fclose(fid);
        end
        
        function plotPopulationEvolution(this)
            % Evolution of the population vs time
            figure;
            plot(this.TD, this.NParICS_alive, 'g-', 'LineWidth', 2, 'DisplayName', 'Alive cells');
            hold on;
            plot(this.TD, this.NParICS_dead, 'r-', 'LineWidth', 2, 'DisplayName', 'Dead cells');
            plot(this.TD, this.NParICS_total, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Total');
            xlabel('Diffusion Time (ms)');
            ylabel('Fraction in ICS');
            title('Population Evolution in Intracellular Space');
            legend('show');
            grid on;
        end
        
        function stats = getPopulationStats(this)
            % Return statistics on populations
            stats.n_alive = this.n_alive;
            stats.n_dead = this.n_dead;
            stats.percent_alive = this.percent_alive;
            stats.percent_dead = this.percent_dead;
            stats.f_alive = this.f_alive;
            stats.kappa_alive = this.kappa_alive;
            stats.kappa_dead = this.kappa_dead;
        end
        
        function dt = dki_shell(this)
            Nt = size(this.sig,1);
            bvecu = unique(this.bvec,'row');
            n2 = this.ndir2(bvecu);
            n4 = this.ndir4(bvecu);
            
            nbvec = size(bvecu,1);
            nbval = numel(unique(this.bval));
            dt = zeros(Nt,21);
            for i = 1:Nt
                sigi = abs(this.sig(i,:));
                Di = zeros(nbvec,1);
                Ki = zeros(nbvec,1);
                for j = 1:nbvec
                    Ij = ismember(this.bvec,bvecu(j,:),'rows');
                    sigj = sigi(Ij); sigj = sigj(:);
                    bvalj = this.bval(Ij); bvalj = bvalj(:);
                    A = [-bvalj 1/6*bvalj.^(2:nbval)];
                    X = A\log(sigj + eps);
                    Di(j) = X(1);
                    Ki(j) = X(2)/X(1)^2;
                end
                dx2g = Di*2.*this.TD(i);
                dx4g = (Ki+3).*dx2g.^2;
                dt(i,1:6) = ( n2\dx2g ).';
                dt(i,7:21) = ( n4\dx4g ).';
            end
        end
                
        function [K,D] = akc_mom(this,n)
            n2 = this.ndir2(n);
            n4 = this.ndir4(n);
            x2 = this.dx2t*n2.';
            x4 = this.dx4t*n4.';
            D = x2/2./this.TD;
            K = x4./x2.^2-3;
        end
        
        function [K,D] = akc_shell(this,dt,n)
            n2 = this.ndir2(n);
            n4 = this.ndir4(n);
            x2 = dt(:,1:6)*n2.';
            x4 = dt(:,7:21)*n4.';
            D = x2/2./this.TD;
            K = x4./x2.^2-3;
        end
        
        function n4i = ndir4(this,ni)
            n4i = prod(reshape(ni(:,this.W_ind),[],15,4),3);
        end
        
        function n2i = ndir2(this,ni)
            n2i = prod(reshape(ni(:,this.D_ind),[],6,2),3);
        end
    end
end