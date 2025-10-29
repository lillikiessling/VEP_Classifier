close all; clc;

% --- Base folder setup ---
input_base = 'Labelled_VEP_Data/PRIMA';
output_base = 'Preprocessed_VEP_Data/PRIMA';

categories = {'BC_and_RGC', 'BC_Only', 'RGC_Only'};

% --- Disable figure visibility (no GUI windows) ---
set(0, 'DefaultFigureVisible', 'off');

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

    for i = 1:length(files)
        [~, file_name, ~] = fileparts(files(i).name);
        filename_str = files(i).name;

        % --- Parse pulse width and signal power from filename ---
        % e.g. PRIMA100_1_10ms_1.39mWmm2.csv
        pulse_width_match = regexp(filename_str, '_(\d+)ms', 'tokens', 'once');
        signal_power_match = regexp(filename_str, '_(\d+\.?\d*)mWmm2', 'tokens', 'once');

        if isempty(pulse_width_match)
            warning('⚠️ Could not parse pulse width from %s. Using default 10ms.', filename_str);
            pulse_width = 10;
        else
            pulse_width = str2double(pulse_width_match{1});
        end

        if isempty(signal_power_match)
            warning('⚠️ Could not parse signal power from %s. Using default 1 mW/mm².', filename_str);
            signal_power = 1;
        else
            signal_power = str2double(signal_power_match{1});
        end

        % --- Process file ---
        try
            [output, snr_noise, snr_signal] = ...
                live_vep_detection_only_tests(input_dir, file_name, signal_power, pulse_width, output_dir, file_name);

            fprintf(' ✅ Processed: %-40s | Signal: %.3f | Noise: %.3f | Power: %.2f mW/mm² | PW: %d ms\n', ...
                file_name, snr_signal, snr_noise, signal_power, pulse_width);

        catch ME
            fprintf(' ❌ Error in %s: %s\n', file_name, ME.message);
        end
    end
end

fprintf('\n✅ All categories processed and saved to %s\n', output_base);
