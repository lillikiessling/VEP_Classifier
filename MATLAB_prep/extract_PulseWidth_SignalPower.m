function [pulseWidth, signalPower] = extract_PulseWidth_SignalPower(filename)
    [~, name, ~] = fileparts(filename);

    % Split the name by underscores
    parts = strsplit(name, '_');

    % Find the part that contains 'ms'
    pulsePart = '';
    pulseIndex = 0;
    for i = 1:length(parts)
        if contains(parts{i}, 'ms', 'IgnoreCase', true)
            pulsePart = parts{i};
            pulseIndex = i;
            break;
        end
    end

    if isempty(pulsePart)
        error('No pulse width information found in filename: %s', filename);
    end


    if pulseIndex == 4 && strcmp(parts{pulseIndex - 1}, '0')
         % Recombine parts, e.g., combine '0' (at index 3) and '5ms' (at index 4)
         pulsePart = [parts{pulseIndex - 1}, '_', parts{pulseIndex}];
    end

    pulsePart = strrep(pulsePart, 'ms', '');  % remove 'ms'
    pulsePart = strrep(pulsePart, '_', '.');  % replace '_' with '.'

    % Convert to numeric
    pulseWidth = str2double(pulsePart);

    if isnan(pulseWidth)
        error('Could not parse pulse width from filename: %s', filename);
    end

    % --- Extract irradiance / signal power ---
    % Look for the last underscore part (should contain 'mWmm2')
    signalPart = parts{end};
    signalPart = strrep(signalPart, 'mWmm2', '');
    signalPower = str2double(signalPart);

    if isnan(signalPower)
        error('Could not parse signal power from filename: %s', filename);
    end
end
