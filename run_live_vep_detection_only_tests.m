close all; clc;

% --- Base folder setup ---
input_base = 'Labelled_VEP_Data/PRIMA';
output_base = 'Preprocessed_VEP_Data/PRIMA';

categories = {'BC_and_RGC', 'BC_Only', 'RGC_Only'};

% --- Disable figure visibility (no GUI windows) ---
set(0, 'DefaultFigureVisible', 'off');

all_summary = table('Size', [0 7], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double', 'string'}, ...
    'VariableNames', {'FileName', 'SNR_Signal', 'SNR_Noise', 'SignalPower_mWmm2', 'PulseWidth_ms', 'SNR_dB', 'Category'});

for c = 1:length(categories)
    category = categories{c};
    input_dir = fullfile(input_base, category);
    output_dir = fullfile(output_base, category);

    % make sure output folder exists
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % find all .csv files in input_dir
    files = dir(fullfile(input_dir, '*.csv'));
    fprintf('Processing %d files in %s...\n', length(files), category);

    summary = table('Size', [0 7], ...
        'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double', 'string'}, ...
        'VariableNames', {'FileName', 'SNR_Signal', 'SNR_Noise', 'SignalPower_mWmm2', 'PulseWidth_ms', 'SNR_dB', 'Category'});


    for i = 1:length(files)
        [~, file_name, ~] = fileparts(files(i).name);
        filename_str = files(i).name;

        % --- Parse pulse width and signal power from filename ---
        % e.g. PRIMA100_1_10ms_1.39mWmm2.csv
        pulse_width_match = regexp(filename_str, '_(\d+)ms', 'tokens', 'once');
        signal_power_match = regexp(filename_str, '_(\d+\.?\d*)mWmm2', 'tokens', 'once');

        if isempty(pulse_width_match)
            warning('Could not parse pulse width from %s. Using default 10ms.', filename_str);
            pulse_width = 10;
        else
            pulse_width = str2double(pulse_width_match{1});
        end

        if isempty(signal_power_match)
            warning('Could not parse signal power from %s. Using default 1 mW/mm².', filename_str);
            signal_power = 1;
        else
            signal_power = str2double(signal_power_match{1});
        end

        % --- Process file ---
        try
            [output, snr_noise, snr_signal] = ...
                live_vep_detection_only_tests(input_dir, file_name, signal_power, pulse_width, output_dir, file_name);
            
            snr_db = 10 * log10(snr_signal / snr_noise); 
            
            fprintf(' Processed: %-40s | Signal: %.3f | Noise: %.3f | Power: %.2f mW/mm² | PW: %d ms\n', ...
                file_name, snr_signal, snr_noise, signal_power, pulse_width);

            summary = [summary; {
                string(file_name), snr_signal, snr_noise, signal_power, pulse_width, snr_db, string(category)
            }];

        catch ME
            fprintf(' Error in %s: %s\n', file_name, ME.message);
        end
    end
    summary_path = fullfile(output_dir, sprintf('SNR_summary_%s.csv', category));
    writetable(summary, summary_path);
    fprintf('Saved summary table: %s\n', summary_path);

    % Append to master table
    all_summary = [all_summary; summary];
end

combined_path = fullfile(output_base, 'SNR_summary_all_categories.csv');
writetable(all_summary, combined_path);
fprintf('\nAll categories processed and combined summary saved to:\n%s\n', combined_path);


fprintf('\nAll categories processed and saved to %s\n', output_base);
