function params = get_sobol_samples(projectpath,Dex_sample,rmean_sample,rsd_sample,f_sample,Ncombos)

    sobol_path = fullfile(projectpath,'sobol_params.txt');
    if ~exist(sobol_path,'file')
        
        disp('Sobol param file not found. Attempting to generate samples...')
        % Save as a txt file
        fileID = fopen(sobol_path,'w');
        fprintf(fileID,sprintf('%u %u\n%u %u\n%u %u\n%u %u\n%u\n',Dex_sample.lb,Dex_sample.ub,rmean_sample.lb,rmean_sample.ub,rsd_sample.lb,rsd_sample.ub,f_sample.lb,f_sample.ub,Ncombos));
        fclose(fileID);
        
        % Run script for generating Sobol samples
        
        cmd_in = 'module load python/3.9.4'; % or whatever python version you need
        [status_out,cmd_out]=system(cmd_in); disp(cmd_out);
        cmd_in = ['cd ' root_helper]; 
        [status_out,cmd_out]=system(cmd_in); disp(cmd_out);
        cmd_in = ['python3 create_sobol_array.py ' sobol_path]; % or whatever python version you need
        [status_out,cmd_out]=system(cmd_in); disp(cmd_out);
        
    else
        disp('Sobol file found. Loading in array...')
        cd(projectpath)
        params = readmatrix('sobol_array.txt');
        % Params order is Dex, rmean, rsd, f
    end

end