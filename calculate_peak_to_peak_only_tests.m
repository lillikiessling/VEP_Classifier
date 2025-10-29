function [peak_to_peak_amplitudes, signal_noise_peak_to_peak_amplitude, noise_rms] = calculate_peak_to_peak_only_tests( ...
    test_data, laser_intensity, normalize, ymin, ymax, max_time, relative_max_time, ...
    noise_detect_time, noise_window, user_title, save_figures, pulse_width, acuity, save_csv, output_dir, output_file)

% get number of tests
number_of_tests = length(test_data);

% initialize output
peak_to_peak_amplitudes = zeros(1, number_of_tests);

% initialize a figure
fig1 = figure('Position', [0 50 400 400]);
title(user_title, 'Interpreter', 'none');
fig_rows = number_of_tests;
fig_cols = 1;

% set polarity
polarity = 1;

% process and plot each of the individual traces
for i = 1:number_of_tests
    
    % create a subplot for the current test
    subplot(fig_rows, fig_cols, i);

    % extract the data for this test
    current_test_data = test_data{i};
    current_laser_intensity = laser_intensity(i);

    % extract the time vector
    time_ms = current_test_data(:,1); 

    % extract the first and third channels and convert from nV to uV
    ch1 = current_test_data(:,2).* 1e-3;
    ch3 = polarity * current_test_data(:,4) .* 1e-3; % CURRENTLY SET TO CHANNEL 3

    %average two phases (500ms chunks)  
    si = 0.5E-3;
    filtFreq = 32;

    % get the number of data points
    N = length(current_test_data(:,1))/2;

    % average the two phases (500ms chunks)
    ch1 = mean(reshape(ch1, [N, 2]), 2);
    ch3 = mean(reshape(ch3, [N, 2]), 2);
    time_ms = time_ms(1:length(time_ms)/2);
    
    % remove artifacts
    if acuity == 1
        % SEPCTRUM RECONSTRUCTION remove carrier freq arti. Reconstruct by interpolating
        f = linspace(0, 1/si, N+1);
        f = f(1:end-1);
        ch3_f = fft(ch3);
        ch3_f_smth = ch3_f;
        for F_AC = [filtFreq:filtFreq:filtFreq*20]
            [~, target_idx] = min(abs(f-F_AC));
            ch3_f_smth = spect_smooth(ch3_f_smth, target_idx);
        end
        ch3_filt = ifft(ch3_f_smth);
        ch3 = ch3_filt;
    end
    [ch3_no_artifacts] = ArtifactRemovalFun(ch1, ch3, pulse_width, 1);

    % detrend the trace
    detrended_ch3 = detrend(ch3); 
    detrended_ch3_no_artifacts = detrend(ch3_no_artifacts);

    % create suplot for each of the tests 
    subplot(number_of_tests, 1, i);

    % plot the detrended trace of the noise trace in red and the rest in black
    ch3 = detrended_ch3_no_artifacts;
    % ch3 = -ch3;
    plot(time_ms, ch3, 'k', 'LineWidth', 1.3); 
    title_string = 'Test';
    
    % get current axes handles
    ax = gca;
    x_limits = ax.XLim;
    y_limits = ax.YLim;

    % add subplot title
    title_string = strcat(title_string, ' (', num2str(round(current_laser_intensity, 3)),' mW/mm^2)');
    %title(title_string);
    hold on;

    % format figure
    if ~isnan(ymin) && ~isnan(ymax)
        ylim([ymin ymax]); 
    end
    xlim([0 500]);
    grid on;
    
    % now, we search for the peaks
    % now cut out the time window in which we expect to see the peaks
    ch3_maxrange = ch3((time_ms > 20) & (time_ms < max_time));
    time_maxrange = time_ms((time_ms > 20) & (time_ms < max_time));
    
    % extract max and min value from data and calculate the amplitude swing between them
    [minval, indmin_in_maxrange] = findpeaks(-ch3_maxrange, 'SortStr', 'descend', NPeaks=1);
    
    % if relative_max_time is not provided, search for global maximum within window
    if isnan(relative_max_time)

        % get max value and its index in the shortened array
        [maxval, indmax_in_maxrange] = findpeaks(ch3_maxrange, 'SortStr', 'descend', NPeaks=1); 

        % get the index of this max value in the original array
        indmin_in_original = find(time_ms == time_maxrange(indmin_in_maxrange));
        indmax_in_original = find(time_ms == time_maxrange(indmax_in_maxrange));

    % else, search for maximum within a relative_max_time from minimum
    else

        % find the time value at indmin
        time_at_indmin = time_maxrange(indmin_in_maxrange);

        % calculate the target time value
        target_time = time_at_indmin + relative_max_time;
        disp(time_at_indmin)

        % extract the data within the specified range
        ch3_maxrange_limited = ch3((time_ms > time_at_indmin) & (time_ms < target_time));
        time_maxrange_limited = time_ms((time_ms > time_at_indmin) & (time_ms < target_time));

        % Find peak within specified time window
        [maxval, indmax_in_limited] = findpeaks(ch3_maxrange_limited, 'SortStr', 'descend', NPeaks=1);

        % get the index of this max value in the original array
        indmin_in_original = find(time_ms == time_maxrange(indmin_in_maxrange));
        indmax_in_original = find(time_ms == time_maxrange_limited(indmax_in_limited));
    end

    % calculate amplitude swing between min and max
    peak_to_peak_amplitudes(i) = maxval + minval;
    
    % mark max and min values on the plot
    plot(time_ms(indmax_in_original), ch3(indmax_in_original), 'b*'); hold on;
    plot(time_ms(indmin_in_original), ch3(indmin_in_original), 'b*'); hold on;
    
    % add labels
    xlabel('Time [ms]');
    ylabel('VEP Amplitude [\muV]');
    % ylim([-10 10]);
    % axis off;
    % line([500 500], [-8 -6], 'Color', 'k');
    % line([400 500], [-8 -8], 'Color', 'k');

    % ------------ Adding noise detection within signal ----------------
    
    if i == 1
        % add range to find noise within signal
        ch3_noiserange = ch3((time_ms > noise_detect_time) & (time_ms < noise_detect_time + noise_window));
        time_noiserange = time_ms((time_ms > noise_detect_time) & (time_ms < noise_detect_time + noise_window));
        
        % Calculate RMS based on noise within signal
        noise_rms = rms(ch3_noiserange);

        % Testing findpeaks instead of min/max ----------
        [trough_amp, trough_ind] = findpeaks(-ch3_noiserange, 'SortStr', 'descend', NPeaks=1);

        indtrough_in_original = find(time_ms == time_noiserange(trough_ind));

        % Find peak in time window after trough
        ch3_peaknoiserange = ch3((time_ms > time_noiserange(trough_ind)) & (time_ms < (time_noiserange(trough_ind) + noise_window)));
        time_noiserange = time_ms((time_ms > time_noiserange(trough_ind)) & (time_ms < (time_noiserange(trough_ind) + noise_window)));
        [peak_amp, peak_ind] = findpeaks(ch3_peaknoiserange, 'SortStr', 'descend', NPeaks=1);
        indpeak_in_original = find(time_ms == time_noiserange(peak_ind));

        hold on;
        plot(time_ms(indpeak_in_original), ch3(indpeak_in_original), 'r*'); hold on;
        plot(time_ms(indtrough_in_original), ch3(indtrough_in_original), 'r*'); hold off;
        
        %xlim([13 500])
        signal_noise_peak_to_peak_amplitude = peak_amp + trough_amp;
    end

end


% normalize amplitude to noise RMS level, if reqested
ylabel_string = 'P2P VEP Amplitude [\muV]';
if normalize == 1
    peak_to_peak_amplitudes = peak_to_peak_amplitudes ./ noise_rms;
    signal_noise_peak_to_peak_amplitude = signal_noise_peak_to_peak_amplitude./noise_rms;
    ylabel_string = 'P2P VEP Amplitude Normalized by Noise RMS [\muV]';
end

% generate second figure for peak-peak plot
fig2 = figure('Position', [500 50 700 500]);

% add a horizontal line to indicate the noise RMS level (first entry)
yline(signal_noise_peak_to_peak_amplitude, 'r--');
hold on;

% add an asterisk for the signals (the other entries)
plot(laser_intensity(2:end), peak_to_peak_amplitudes(1:end), '*-', 'MarkerSize',10, 'LineStyle', 'none')
hold on;

% format figure
xlabel('Irradiance [mW/mm^2]');
ylabel(ylabel_string);
grid on;
xlim([-1 laser_intensity(end)+3]);
ylim([0 max(peak_to_peak_amplitudes) + 10]);
title(user_title, 'Interpreter', 'none');
set(findall([fig1, fig2], '-property', 'FontName'), 'FontName', 'Times New Roman', 'FontSize', 15);

if save_figures == 1
    saveas(fig1, 'VEP signals', 'jpeg');
    saveas(fig2, ' P2P Amplitude', 'jpeg');
end

if save_csv == 1
    % --- Create output directory if it doesn't exist ---
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % --- Save all processed traces ---
    for i = 1:number_of_tests
        current_test_data = test_data{i};
        time_ms = current_test_data(:,1);

        % Extract channels
        ch1 = current_test_data(:,2) .* 1e-3;
        ch3 = polarity * current_test_data(:,4) .* 1e-3;

        % Average two 500 ms phases
        N = length(current_test_data(:,1)) / 2;
        ch3 = mean(reshape(ch3, [N, 2]), 2);
        time_ms = time_ms(1:length(time_ms)/2);

        % Artifact removal + detrend
        [ch3_no_artifacts] = ArtifactRemovalFun(ch1(1:length(ch3)), ch3, pulse_width, 1);
        ch3 = detrend(ch3_no_artifacts);

        % Combine time and processed channel
        processed_data = [time_ms, ch3];

        % Construct file path
        if number_of_tests == 1
            output_path = fullfile(output_dir, [output_file, '.csv']);
        else
            output_path = fullfile(output_dir, sprintf('%s_test_%d.csv', output_file, i));
        end

        % Write CSV
        writematrix(processed_data, output_path);
    end

    disp(['Saved processed trace(s) to: ', output_dir]);
end


end
