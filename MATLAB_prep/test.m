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

    % find all .csv files in input_dir
    files = dir(fullfile(input_dir, '*.csv'));
    fprintf('Processing %d files in %s...\n', length(files), category);

    for i = 1:length(files)
        filename_str = files(i).name;

        [pulse_width, signal_power] = extract_PulseWidth_SignalPower(filename_str); 

        fprintf('File: %-40s  ->  Pulse Width: %.2f ms | Signal Power: %.2f mW/mm²\n', ...
            filename_str, pulse_width, signal_power);
    end
end
