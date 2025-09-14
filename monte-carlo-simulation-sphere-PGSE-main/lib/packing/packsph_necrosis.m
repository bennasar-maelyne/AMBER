classdef packsph_necrosis < handle
    % Packing spheres with mixed populations (alive/dead cells)
    % 
    % Usage:
    % pc = packsph();
    % pc.packing(maxdensity,hardwallBC,rinit,root,target,compileFlag,n,gap,cellType,kappa_values)
    %
    % Input:
    % maxdensity: maximal sphere volume fraction
    % hardwallBC: boundary condition, 0 for periodic, 1 for hard wall
    % rinit: sphere radius, [# spheres x 1], micron
    % root: directory to Donev's C code
    % target: directory saving the packing
    % compileFlag: 0 for not compiling the C code, 1 for compiling the C code
    % n: size of the lookup table = n x n x n
    % gap: the shortest distance between spheres, micron
    % cellType: cell type array [# spheres x 1], 0=alive, 1=dead
    % kappa_values: permeability values [# spheres x 1], micron/ms
    % 
    % Output:
    % packing_parameter.mat
    % xc, yc, zc, rc: position of the sphere center and the sphere radius
    % within a normalized box
    % res: length of the box side in micron
    % cellType: cell type for each sphere
    % kappa_values: permeability for each sphere
    %
    % lookup_table.mat
    % A: lookup table
    % n: size of the lookup table
    % Nmax: the smallest integer larger than # sphere, in the base of 10
    %
    % phantom_*.txt
    % input files for cuda code
    %
    % Modified version to handle mixed cell populations
    
    properties (Constant = true, Access = protected)
        
    end
    
    properties (GetAccess = public, SetAccess = protected)
        
    end
    
    properties (GetAccess = private, SetAccess = protected)
        
    end
    
    methods (Access = public)
        function this = packsph_necrosis()
            
        end
        
        function packing(this,maxdensity,hardwallBC,rinit,root,target,compileFlag,n,gap,varargin)
            % Handle optional parameters for mixed populations
            if nargin >= 10
                cellType = varargin{1};
                kappa_values = varargin{2};
                mixedPopulation = true;
            else
                cellType = [];
                kappa_values = [];
                mixedPopulation = false;
            end
            
            % Initial position and rescaled outer radius
            rinit_adj = rinit+gap/2;
            maxdensity_adj = maxdensity*sum(rinit_adj.^3)/sum(rinit.^3);
            this.initposition(rinit_adj,root);
            N = length(rinit_adj);
            
            % Prepare input file for C++
            C = cell(1,15);
            fid = fopen(fullfile(root,'spheres_poly/input'),'r');
            for k = 1:15, C{k} = fgetl(fid); end
            fclose(fid);

            % Change N and maxpf variables in inputN file 
            C{2}=['int N = ' num2str(N) ';                        // number of spheres'];
            C{4}=['double maxpf = ' num2str(maxdensity_adj) ';                  // max packing fraction'];
            C{12}=['int hardwallBC = ' num2str(hardwallBC) ';                   // 0 for periodic, 1 for hard wall BC'];
            fid = fopen(fullfile(root,'spheres_poly/inputN'), 'w');
            fprintf(fid, '%s\n', C{:});
            fclose(fid);
            
            % compile and run Donev's C code
            root_tmp = pwd;
            cd(fullfile(root,'spheres_poly'));
            if compileFlag
                disp('compiling .............');
                system('g++ -O3 -ffast-math -o spheres neighbor.C spheres.C box.C sphere.C event.C heap.C read_input.C');
            end
            disp('running C ...........');
            tic;
            system('./spheres inputN'); % run
            toc;
            cd(root_tmp);
            
            % Save final packing
            outputfilename = fullfile(root,'spheres_poly/write.dat');
            M = dlmread(outputfilename,'',6,0);
            xc = M(:,1); yc = M(:,2); zc = M(:,3); rc = M(:,4)/2;
            
            % Adjust radius for targeted density
            density = sum(4/3*pi*rc.^3);
            if density>maxdensity
                rc = (maxdensity_adj/density)^(1/3)*rc;
            end
            
            % Calculate field-of-view in micron
            res = mean(rinit_adj)/mean(rc);
            rc = rc-gap/res/2;
            
            fprintf('Sphere volume fraction = %.4f\n',sum(4/3*pi*rc.^3));
            
            % Create lookup table (enhanced for mixed populations)
            if mixedPopulation
                [A,~,Nmax] = this.createlookuptable_mixed(n,xc,yc,zc,rc,cellType);
            else
                [A,~,Nmax] = this.createlookuptable(n,xc,yc,zc,rc);
            end
            
            % Save files for C++ simulation code
            fid = fopen(fullfile(target,'phantom_res.txt'),'w');
            fprintf(fid,'%f',res);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_NPix.txt'),'w');
            fprintf(fid,'%u',n);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_APix.txt'),'w');
            for i = 1:size(A,1)
                for j = 1:size(A,2)
                    fprintf(fid,sprintf('%u\n',A(i,j,:)));
                end
            end
            fclose(fid);
            
            fid = fopen(fullfile(target,'phantom_NAx.txt'),'w');
            fprintf(fid,'%u',length(rc));
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_xCir.txt'),'w');
            fprintf(fid,'%f\n',xc);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_yCir.txt'),'w');
            fprintf(fid,'%f\n',yc);
            fclose(fid);
            
            fid = fopen(fullfile(target,'phantom_zCir.txt'),'w');
            fprintf(fid,'%f\n',zc);
            fclose(fid);
            
            fid = fopen(fullfile(target,'phantom_rCir.txt'),'w');
            fprintf(fid,'%f\n',rc);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_Nmax.txt'),'w');
            fprintf(fid,'%u',Nmax);
            fclose(fid);
            
            % Save mixed population data if provided
            if mixedPopulation
                % Save cell types
                fid = fopen(fullfile(target,'phantom_cellType.txt'),'w');
                fprintf(fid,'%d\n',cellType);
                fclose(fid);
                
                % Save permeability values
                fid = fopen(fullfile(target,'phantom_kappa.txt'),'w');
                fprintf(fid,'%.6f\n',kappa_values);
                fclose(fid);
                
                % Save population statistics
                n_alive = sum(cellType == 0);
                n_dead = sum(cellType == 1);
                
                fid = fopen(fullfile(target,'phantom_population_stats.txt'),'w');
                fprintf(fid,'Total spheres: %d\n', length(cellType));
                fprintf(fid,'Alive cells: %d (%.1f%%)\n', n_alive, 100*n_alive/length(cellType));
                fprintf(fid,'Dead cells: %d (%.1f%%)\n', n_dead, 100*n_dead/length(cellType));
                if n_alive > 0
                    fprintf(fid,'Mean radius alive: %.4f um\n', mean(rc(cellType==0)));
                    fprintf(fid,'Std radius alive: %.4f um\n', std(rc(cellType==0)));
                    fprintf(fid,'Mean kappa alive: %.6f um/ms\n', mean(kappa_values(cellType==0)));
                end
                if n_dead > 0
                    fprintf(fid,'Mean radius dead: %.4f um\n', mean(rc(cellType==1)));
                    fprintf(fid,'Std radius dead: %.4f um\n', std(rc(cellType==1)));
                    fprintf(fid,'Mean kappa dead: %.6f um/ms\n', mean(kappa_values(cellType==1)));
                end
                fclose(fid);
                
                % Save extended packing parameters
                save(fullfile(target,'packing_parameter_mixed.mat'),'xc','yc','zc','rc','res','cellType','kappa_values');
            else
                % Save standard packing parameters
                save(fullfile(target,'packing_parameter.mat'),'xc','yc','zc','rc','res');
            end
        end
        
        function arrange(this,target,xc,yc,zc,rc,res,n,varargin)
            % Handle optional parameters for mixed populations
            if nargin >= 9
                cellType = varargin{1};
                kappa_values = varargin{2};
                mixedPopulation = true;
            else
                cellType = [];
                kappa_values = [];
                mixedPopulation = false;
            end
            
            if mixedPopulation
                [A,~,Nmax] = this.createlookuptable_mixed(n,xc,yc,zc,rc,cellType);
            else
                [A,~,Nmax] = this.createlookuptable(n,xc,yc,zc,rc);
            end
            
            fid = fopen(fullfile(target,'phantom_res.txt'),'w');
            fprintf(fid,'%f',res);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_NPix.txt'),'w');
            fprintf(fid,'%u',n);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_APix.txt'),'w');
            for i = 1:size(A,1)
                for j = 1:size(A,2)
                    fprintf(fid,sprintf('%u\n',A(i,j,:)));
                end
            end
            fclose(fid);
            
            fid = fopen(fullfile(target,'phantom_NAx.txt'),'w');
            fprintf(fid,'%u',length(rc));
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_xCir.txt'),'w');
            fprintf(fid,'%f\n',xc);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_yCir.txt'),'w');
            fprintf(fid,'%f\n',yc);
            fclose(fid);
            
            fid = fopen(fullfile(target,'phantom_zCir.txt'),'w');
            fprintf(fid,'%f\n',zc);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_rCir.txt'),'w');
            fprintf(fid,'%f\n',rc);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_Nmax.txt'),'w');
            fprintf(fid,'%u',Nmax);
            fclose(fid);
            
            % Save mixed population data if provided
            if mixedPopulation
                fid = fopen(fullfile(target,'phantom_cellType.txt'),'w');
                fprintf(fid,'%d\n',cellType);
                fclose(fid);
                
                fid = fopen(fullfile(target,'phantom_kappa.txt'),'w');
                fprintf(fid,'%.6f\n',kappa_values);
                fclose(fid);
            end
        end
        
        function [A,B,Nmax] = createlookuptable_mixed(this,n,xc,yc,zc,r,cellType)
        %CREATELOOKUPTABLE_MIXED    create lookup table for packed spheres with cell types
        %   [A,B,Nmax] = createlookuptable_mixed(n,xc,yc,zc,r,cellType)
        %   
        %   Input:
        %   n: size of the lookup table = n x n x n
        %   xc, yc, zc: center of spheres, 0 <= xc,yc,zc <= 1
        %   r: outer radius of spheres
        %   cellType: cell type for each sphere (0=alive, 1=dead)
        %
        %   Output:
        %   A: sphere labels/lookup table (includes cell type information)
        %   B: # spheres in each pixel
        %   Nmax: the smallest integer larger than # spheres, in the base of 10
        %
        %   Modified to encode cell type information in the lookup table
        
            MSX = n;
            MSY = n;
            MSZ = n;
            A = zeros(MSX,MSY,MSZ);
            B = zeros(MSX,MSY,MSZ);
            N = length(r);
            Nmax = 10^(ceil(log10(N+1)));

            for i = 1:size(r,1)
                for ii = ceil((xc(i)-r(i))*n):(ceil((xc(i)+r(i))*n)+1)
                    if (ii>MSX)
                       ti = ii-MSX;
                    elseif (ii < 1)
                       ti = ii+MSX;
                    else
                       ti = ii;
                    end
                    for jj = ceil((yc(i)-r(i))*n):(ceil((yc(i)+r(i))*n)+1)
                        if jj > MSY
                            tj = jj-MSY;
                        elseif (jj < 1)
                            tj = jj+MSY;
                        else
                            tj = jj;
                        end
                        for kk = ceil((zc(i)-r(i))*n):(ceil((zc(i)+r(i))*n)+1)
                            if kk > MSZ
                                tk = kk-MSZ;
                            elseif (kk < 1)
                                tk = kk+MSZ;
                            else
                                tk = kk;
                            end
                            if this.inside_sphere(ii,jj,kk,xc(i),yc(i),zc(i),r(i),n)
                                if A(ti,tj,tk) == 0        % 1 sphere
                                    A(ti,tj,tk) = i;
                                    B(ti,tj,tk) = 1;
                                else
                                    if A(ti,tj,tk) < Nmax  % 2 spheres
                                        A(ti,tj,tk) = A(ti,tj,tk)*Nmax + i;
                                        B(ti,tj,tk) = 2;
                                    else                   % > 2 spheres
                                        fprintf('More than 2 spheres in one pixel.\n');
                                        fprintf('Use larger matrix size for the table.\n');
                                        pause;
                                    end
                                end                    
                            end
                        end
                    end
                end
            end
            A = uint32(A);
            B = uint16(B);
            fprintf(' * Matrix filling done ! *\n');
            fprintf(' * Mixed population lookup table created *\n');
            fprintf(' ----------------------------\n');
        end
        
        function [A,B,Nmax] = createlookuptable(this,n,xc,yc,zc,r)
        %CREATELOOKUPTABLE    create lookup table for packed spheres
        %   [A,B,Nmax] = createlookuptable(n,xc,yc,zc,r)
        %   
        %   Input:
        %   n: size of the lookup table = n x n x n
        %   xc, yc, zc: center of spheres, 0 <= xc,yc,zc <= 1
        %   r: outer radius of cylinders
        %
        %   Output:
        %   A: axon labels/lookup table
        %   B: # axon in each pixel
        %   Nmax: the smallest integer larger than # axon, in the base of 10
        %
        %   Author: Hong-Hsi Lee, 2016 (orcid.org/0000-0002-3663-6559)
        %           Lauren M Burcaw, 2014
            MSX = n;
            MSY = n;
            MSZ = n;
            A = zeros(MSX,MSY,MSZ);
            B = zeros(MSX,MSY,MSZ);
            N = length(r);
            Nmax = 10^(ceil(log10(N+1)));

            for i = 1:size(r,1)
                for ii = ceil((xc(i)-r(i))*n):(ceil((xc(i)+r(i))*n)+1)
                    if (ii>MSX)
                       ti = ii-MSX;
                    elseif (ii < 1)
                       ti = ii+MSX;
                    else
                       ti = ii;
                    end
                    for jj = ceil((yc(i)-r(i))*n):(ceil((yc(i)+r(i))*n)+1)
                        if jj > MSY
                            tj = jj-MSY;
                        elseif (jj < 1)
                            tj = jj+MSY;
                        else
                            tj = jj;
                        end
                        for kk = ceil((zc(i)-r(i))*n):(ceil((zc(i)+r(i))*n)+1)
                            if kk > MSZ
                                tk = kk-MSZ;
                            elseif (kk < 1)
                                tk = kk+MSZ;
                            else
                                tk = kk;
                            end
                            if this.inside_sphere(ii,jj,kk,xc(i),yc(i),zc(i),r(i),n)
                                if A(ti,tj,tk) == 0        % 1 sphere
                                    A(ti,tj,tk) = i;
                                    B(ti,tj,tk) = 1;
                                else
                                    if A(ti,tj,tk) < Nmax  % 2 spheres
                                        A(ti,tj,tk) = A(ti,tj,tk)*Nmax + i;
                                        B(ti,tj,tk) = 2;
                                    else                   % > 2 spheres
                                        fprintf('More than 2 spheres in one pixel.\n');
                                        fprintf('Use larger matrix size for the table.\n');
                                        pause;
                                    end
                                end                    
                            end
                        end
                    end
                end
            end
            A = uint32(A);
            B = uint16(B);
            fprintf(' * Matrix filling done ! *\n');
            fprintf(' ----------------------------\n');
        end
        
        function plotpack_mixed(this,xc,yc,zc,rc,cellType)
        %PLOTPACK_MIXED    Plot sphere packing with cell type visualization
        %   plotpack_mixed(xc,yc,zc,rc,cellType) plots sphere packing with
        %   different colors for alive (green) and dead (red) cells
            
            figure;
            hold on;
            [xs, ys, zs] = sphere;
            
            % Plot alive cells in green
            alive_indices = find(cellType == 0);
            for i = alive_indices'
                surf(xs*rc(i)+xc(i), ys*rc(i)+yc(i), zs*rc(i)+zc(i), ...
                     'edgealpha', 0, 'facecolor', [0 0.8 0], 'facealpha', 0.7);
            end
            
            % Plot dead cells in red  
            dead_indices = find(cellType == 1);
            for i = dead_indices'
                surf(xs*rc(i)+xc(i), ys*rc(i)+yc(i), zs*rc(i)+zc(i), ...
                     'edgealpha', 0, 'facecolor', [0.8 0 0], 'facealpha', 0.7);
            end
            
            xlim([0 1]); ylim([0 1]); zlim([0 1]);
            pbaspect([1 1 1]); box on
            title(sprintf('Mixed Cell Population Packing\nAlive: %d, Dead: %d', ...
                  sum(cellType==0), sum(cellType==1)), 'interpreter', 'latex', 'fontsize', 16)
            set(gca,'xtick',[],'ytick',[],'ztick',[])
            view(3)
            material dull
            camlight
            
            % Add legend
            h1 = surf(NaN(2), NaN(2), NaN(2), 'facecolor', [0 0.8 0], 'facealpha', 0.7);
            h2 = surf(NaN(2), NaN(2), NaN(2), 'facecolor', [0.8 0 0], 'facealpha', 0.7);
            legend([h1, h2], {'Alive cells', 'Dead cells'}, 'Location', 'best');

            
            hold off;
        end
    end

    methods (Static)
        function inside = inside_sphere(ii,jj,kk,xc,yc,zc,r,n)
        %INSIDE_SPHERE    True for the pixel overlapping the sphere
        %   inside_sphere(ii,jj,kk,xc,yc,zc,r,n) returns 1 if the pixel (ii,jj,kk)
        %   overlaps with the sphere (xc,yc,zc,r), otherwise, return 0. The size of
        %   the lookup table is n x n x n.
        %
        %   Author: Hong-Hsi Lee, 2016 (orcid.org/0000-0002-3663-6559)
        
        x = max( ii-1, min(xc*n,ii) );
        y = max( jj-1, min(yc*n,jj) );
        z = max( kk-1, min(zc*n,kk) );
        
        d2 = (x-xc*n)^2 + (y-yc*n)^2 + (z-zc*n)^2;
        inside = d2 < (r*n)^2;
        end
        
        function [x, y, z, rs] = initposition(rinit,root) 
        %INITPOSITION    Initialize positions and radii of densly packed spheres
        %   [x,y,z,rs] = initposition(rinit) returns initial positions x, y, z and
        %   rescaled radii rs of spheres for Donev's C++ input file, based on
        %   initial radii rinit (column vector), and saves them in the file
        %   specified in readfilename.
        %
        %   ---------------------------------------------------------
        %     in box.C, in the function
        %     void box::ReadPositions(const char* filename)
        %
        %   infile.ignore(256, '\n');  // ignore the dim line
        %   infile.ignore(256, '\n');  // ignore the #sphere 1 line
        %   infile.ignore(256, '\n');  // ignore the #sphere line
        %   infile.ignore(256, '\n');  // ignore the diameter line
        %   infile.ignore(1000, '\n'); // ignore the 100 010 001 line
        %   infile.ignore(256, '\n');  // ignore the T T T line
        % 
        %   for (int i=0; i<N; i++)
        %     {
        %       infile >> s[i].r;      // read in radius    
        %       infile >> s[i].gr;     // read in growth rate
        %       infile >> s[i].m;      // read in mass
        %       for (int k=0; k<DIM; k++)  
        %          infile >> s[i].x[k]; // read in position 
        %     }
        %    ..... 
            readfilename=fullfile(root,'spheres_poly/read.dat');
            dim=3;              % dimension
            N=length(rinit);    % # cylinders

            % rescale radii to make them small enough to not overlap, but not too small
            rmax=max(rinit); 
            dens0=0.01;         % dens0 ~ N(rmax/Rscale)^3
            % divide all radii by Rscale for less initial overlap
            Rscale=rmax*(N/dens0)^(1/3); 
            rinit=sort(rinit(:),'descend');
            rs=rinit/Rscale;

            % assign random positions and check for no overlap
            x=zeros(N,1); y=zeros(N,1); z=zeros(N,1); 
            % distance^2 function
            dist2 = @(x1,y1,z1, x2,y2,z2) min(abs(x1-x2),1-abs(x1-x2))^2 + min(abs(y1-y2),1-abs(y1-y2))^2 + min(abs(z1-z2),1-abs(z1-z2))^2;
            for ncurr=1:N
               overlap=1;
               while (overlap==1)
                  overlap=0;
                  x(ncurr)=rand; y(ncurr)=rand; z(ncurr)=rand;
                  for nprev=1:ncurr-1
                     if dist2(x(ncurr),y(ncurr),z(ncurr),x(nprev),y(nprev),z(nprev))<=(rs(ncurr)+rs(nprev))^2, overlap=1; ncurr; break, end
                  end
               end
            end

            % create read.dat file
            fid=fopen(readfilename,'w');
            fprintf(fid,'%d\n', dim);
            fprintf(fid,'%d\n', N);
            fprintf(fid,'%d\n', N);
            fprintf(fid,'%e\n', 2*max(rs));
            fprintf(fid,'10 01\n');
            fprintf(fid,'T T\n');
            for n=1:N
               fprintf(fid, '%e %e %f %f %f\n', rs(n), rs(n), 1.0, x(n), y(n), z(n));
            end
            fclose(fid);
        end
        
        function plothist(rinit,nbin)
        %PLOTHIST    Plot diameter histogram
        %   plothist(rinig,nbin) plots diameter histogram based on the
        %   radii rinit, and number of bins nbin.
            diam = 2 * rinit; % µm
            edges = linspace(min(diam), max(diam), nbin+1);
            Nc = histcounts(diam, edges);
            Nc = Nc / sum(Nc) / mean(diff(edges));

            % Plot
            bar(edges(1:end-1) + diff(edges)/2, Nc, 1); % center
            box on; 
            pbaspect([2 1 1])
            xlabel('Inner Diameter ($\mu$m)','interpreter','latex','fontsize',20)
            ylabel('PDF ($\mu$m$^{-1}$)','interpreter','latex','fontsize',20)
        end
        
        function plothist_mixed(rinit,cellType,nbin)
        %PLOTHIST_MIXED    Plot diameter histogram for mixed populations
        %   plothist_mixed(rinit,cellType,nbin) plots diameter histogram 
        %   separately for alive and dead cells
            
            rinit_alive = rinit(cellType == 0);
            rinit_dead = rinit(cellType == 1);
            
            figure;
            hold on;
            
            if ~isempty(rinit_alive)
                edges_alive = linspace(0, max(rinit_alive), nbin);
                Nc_alive = histcounts(2*rinit_alive, edges_alive);
                Nc_alive = Nc_alive/sum(Nc_alive)/mean(diff(edges_alive));
                bar(edges_alive(2:end), Nc_alive, 1, 'FaceColor', 'green', ...
                    'FaceAlpha', 0.6, 'DisplayName', 'Alive cells');
            end
            
            if ~isempty(rinit_dead)
                edges_dead = linspace(0, max(rinit_dead), nbin);
                Nc_dead = histcounts(2*rinit_dead, edges_dead);
                Nc_dead = Nc_dead/sum(Nc_dead)/mean(diff(edges_dead));
                bar(edges_dead(2:end), Nc_dead, 1, 'FaceColor', 'red', ...
                    'FaceAlpha', 0.6, 'DisplayName', 'Dead cells');
            end
            
            box on; pbaspect([2 1 1])
            xlabel('Inner Diameter ($\mu$m)','interpreter','latex','fontsize',20)
            ylabel('PDF ($\mu$m$^{-1}$)','interpreter','latex','fontsize',20)
            legend('Location', 'best');
            title('Cell Diameter Distribution by Population', 'interpreter', 'latex', 'fontsize', 16);
            hold off;
        end
        
        function plotpack(xc,yc,zc,rc)
        %PLOTPACK    Plot cylinder packing
        %   plotpack(xc,yc,rc) plots cylinder packing based on the cylinder
        %   positions xc, yc and radii rc. Positions and radii are rescaled
        %   to fit within a 1 x 1 square.
            hold on;
            [xs, ys, zs] = sphere;
            for ii = 0%-1:1
                for jj = 0%-1:1
                    for kk = 0%-1:1
                        for i = 1:numel(rc)
                            surf(xs*rc(i)+xc(i)+ii, ys*rc(i)+yc(i)+jj, zs*rc(i)+zc(i)+kk,'edgealpha',0,'facecolor',0.7*[1 1 1]);
                        end
                    end
                end
            end
            xlim([0 1]); ylim([0 1]); zlim([0 1]);
            pbaspect([1 1 1]); box on
%             title('Sphere Packing','interpreter','latex','fontsize',20)
            set(gca,'xtick',[],'ytick',[],'ztick',[])
            view(3)
            material dull
            camlight
        end
        
        function plotlookup (A,Nsph,sl)
        %PLOTLOOKUP    Plot lookup table
        %   plotlookup(A,Nsph,sl) plots the lookup table A of Nsph packed 
        %   spheres on the sl-th slice. The background is black, and the 
        %   pixels with two spheres are white.
            cmap = colormap('parula');
            Ibg = A==0;                     % background region
            Iol = A>Nsph;                   % two-axon region
            A2 = ceil(single(A)/Nsph*64);   % rescale the colormap for # axons
            A2(Ibg) = 1;
            A2(Iol) = 1;
            [nx,ny,nz] = size(A2);
            imgc = cmap(uint16(A2(:)),:);
            imgc(Ibg,:) = 0;                % background region is black
            imgc(Iol,:) = 1;                % two-axon region is white
            imgc = reshape(imgc,[nx,ny,nz,3]);

            image(rot90(squeeze(imgc(:,:,sl,:)))); caxis([0 Nsph]);
            box on; axis off; pbaspect([1 1 1]);
            title('Lookup Table','interpreter','latex','fontsize',20)
        end
        
        function plotlookup_mixed(A,Nsph,sl,cellType)
        %PLOTLOOKUP_MIXED    Plot lookup table with cell type information
        %   plotlookup_mixed(A,Nsph,sl,cellType) plots the lookup table A 
        %   with different colors for alive and dead cell regions
            
            cmap_alive = [0 1 0; 0 0.8 0; 0 0.6 0; 0 0.4 0]; % green shades
            cmap_dead = [1 0 0; 0.8 0 0; 0.6 0 0; 0.4 0 0];  % red shades
            
            A_slice = A(:,:,sl);
            [nx,ny] = size(A_slice);
            imgc = zeros(nx,ny,3);
            
            % Color pixels based on cell type
            for i = 1:nx
                for j = 1:ny
                    pixel_val = A_slice(i,j);
                    if pixel_val == 0
                        imgc(i,j,:) = [0 0 0]; % background is black
                    else
                        if pixel_val <= Nsph
                            cell_idx = pixel_val;
                        else
                            cell_idx = mod(pixel_val, Nsph);
                            if cell_idx == 0, cell_idx = Nsph; end
                        end
                        
                        if cell_idx <= length(cellType)
                            if cellType(cell_idx) == 0 % alive
                                imgc(i,j,:) = cmap_alive(1,:);
                            else % dead
                                imgc(i,j,:) = cmap_dead(1,:);
                            end
                        else
                            imgc(i,j,:) = [0.5 0.5 0.5]; % unknown
                        end
                    end
                end
            end
            
            image(rot90(imgc));
            box on; axis off; pbaspect([1 1 1]);
            title('Mixed Population Lookup Table','interpreter','latex','fontsize',16)
        end
        
        function rinit = gaussdist(N,rmean,rstd,seed)
        %GAUSSDIST    Create Gaussian distributed radii
        %   rinit=gaussdist(N,rmean,rstd,seed) creates Gaussian distributed
        %   cylindrical radii rinit with mean rmean and standard deviation
        %   rstd. The output has N radii, and the randn is initilized by
        %   seed.
            rng(seed);
            rinit = [];
            while(numel(rinit)<N)
                ri = rmean + rstd*randn(N,1);
                ri = ri(ri>0);
                rinit = cat(1,rinit,ri(:));
            end
            rinit = rinit(1:N);
            rinit = rinit/mean(rinit)*rmean;
            rinit = sort(rinit);
        end
        
        function rinit = anydist(N,frequency,ri)
        %ANYDIST    Create radii based on a given distribution
        %   rinit=anydist(N,frequency,ri) creates cylindrical radii rinit 
        %   based on any given discretized distribution, frequency(ri). The
        %   output has N radii.
            pct = frequency/sum(frequency);         % frequency, count in percentage
            Nct = round(N*pct);                     % actual count
            if sum(Nct)<N
                I = randperm(length(Nct),N-sum(Nct));
                Nct(I) = Nct(I) + 1;
            elseif sum(Nct)>N
                Ip = find(Nct>0);
                I = randperm(numel(Ip),sum(Nct)-N);
                Nct(Ip(I)) = Nct(Ip(I)) - 1;
            end
            
            rinit = zeros(N,1);                     % output radius
            i = 0;
            for bin = 1:length(Nct)
              rinit(i+1:i+Nct(bin)) = ri(bin);
              i = i+Nct(bin);
            end
        end
        
        function ds = sphgap(xc,yc,zc,rc)
            xc = xc(:); yc = yc(:); zc = zc(:); rc = rc(:);
            ds = sqrt((xc-xc.').^2 + (yc-yc.').^2 + (zc-zc.').^2) - (rc+rc.');
            n = length(rc);
            ds(1:n+1:end) = 0;
        end
        
    end
    
end