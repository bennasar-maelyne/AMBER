function params = get_sobol_samples_V2(projectpath,root_helper,Dex_sample,rmean_sample,rsd_sample,f_sample,Ncombos)

    sobol_path = fullfile(projectpath,'sobol_params.txt');
    sobol_arraypath = fullfile(projectpath,'sobol_array.txt');

    if ~exist(sobol_arraypath,'file')
        
        disp('Sobol array file not found. Attempting to generate samples...')
        % Save as a txt file
        fileID = fopen(sobol_path,'w');
        %fprintf(fileID,sprintf('%u %u\n%u %u\n%u %u\n%u %u\n%u %u\n%u',Dex_sample.lb,Dex_sample.ub,rmean_sample.lb,rmean_sample.ub,rsd_sample.lb,rsd_sample.ub,f_sample.lb,f_sample.ub,Delta_sample.lb,Delta_sample.ub,Ncombos));
        fprintf(fileID,sprintf('%u %u\n%u %u\n%u %u\n%u %u\n%u %u\n%u',Dex_sample.lb,Dex_sample.ub,rmean_sample.lb,rmean_sample.ub,rsd_sample.lb,rsd_sample.ub,f_sample.lb,f_sample.ub,Ncombos));
        fclose(fileID);
        
        % Run script for generating Sobol samples
        % 
        disp('Please copy and paste the following commands in the terminal:')
        disp('module load python/3.9.4')
        disp(['cd ' root_helper])
        disp(['python3 create_sobol_array.py ' sobol_path])
%         cmd_in = 'module load python/3.9.4'; % or whatever python version you need
%         [status_out,cmd_out]=system(cmd_in); disp(cmd_out);
%         cmd_in = ['cd ' root_helper]; 
%         [status_out,cmd_out]=system(cmd_in); disp(cmd_out);
%         cmd_in = ['python3 create_sobol_array.py ' sobol_path]; % or whatever python version you need
%         [status_out,cmd_out]=system(cmd_in); disp(cmd_out);
        input('Press enter when finished:')
        try
            fullfile(projectpath)
            params = readmatrix('sobol_array.txt');
            disp('Sobol params successfully loaded.')
        catch
            disp('Sobol params not loaded!');
        end

        
    else
        disp('Sobol file found. Loading in array...')
        fullfile(projectpath)
        params = readmatrix('sobol_array.txt');
        % Params order is Dex, rmean, rsd, f
    end

end