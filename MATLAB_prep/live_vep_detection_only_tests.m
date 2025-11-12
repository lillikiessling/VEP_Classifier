function [output_msg, signal_noise_peak_to_peak, signal_peak_to_peak]= live_vep_detection_only_tests(working_directory, file_name, signal_power,pulse_width, output_dir, output_file)
    
    % initialize output
    output_msg = "";

    % verify provided path 
    verify_working_directory(working_directory);

    %% find the most recent folder in the working directory (should contain the latest measurement)
    
    % get most recent experiment folder
    [latest_folder, ~] = get_most_recent_item(working_directory);
    
    % print the name of the latest folder
    msg = sprintf('Measurement folder: %s\n', latest_folder);
    output_msg = [output_msg, msg];
    
    % set full path for further processing
    latest_folder = fullfile(working_directory, latest_folder);

    number_of_tests = 1; %length(test_list);
    
    % initialize raw data output 
    raw_data = cell(1, number_of_tests);
    
    % extrat the raw data for the obtained test files
    for i = 1:number_of_tests

        % generate full path
        test_file = fullfile(latest_folder,strcat(file_name, '.csv'));
        
        % read test data and store it 
        raw_data{i} = csvread(test_file, 3, 6);
        
    end    
    
    % set processing parameters
    normalize_to_noise = false;
    laser_intensity = [0, signal_power];
    y_min = nan;
    y_max = nan;
    pk_to_pk_time_window = 120;
    inner_max_window = 120;
    noise_detect_time = 400;
    noise_window = 50;
    user_title = "";
    save_figures = false;
    acuity = 0;
    save_data_to_csv = 1; 
    
    % calculate peak-to-peak normalized to noise level
    [peak_to_peak_amplitudes, signal_noise_peak_to_peak_amplitude, ~] = calculate_peak_to_peak_only_tests(raw_data, laser_intensity, normalize_to_noise, y_min, y_max, pk_to_pk_time_window, inner_max_window, noise_detect_time, noise_window, user_title, save_figures, pulse_width, acuity, save_data_to_csv, output_dir, output_file);
    % extract signal peak-to-peak
    signal_peak_to_peak = peak_to_peak_amplitudes(1);
    signal_noise_peak_to_peak = signal_noise_peak_to_peak_amplitude;
    
    % check whether the measured signal is above noise level and report
    if (signal_peak_to_peak > signal_noise_peak_to_peak*1.1)
        msg = sprintf('Measured signal is above noise level.\nPeak-to-peak normalized to noise is: %duV\n', signal_peak_to_peak);
    else
        msg = sprintf('\nNo response detected.\nPeak-to-peak normalized to noise RMS is: %.5fuV\n', round(signal_peak_to_peak, 5));
    end
    output_msg = [output_msg, msg];

    % concat all to one output message 
    output_msg = strjoin(output_msg, '');

end