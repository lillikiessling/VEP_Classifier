function [most_recent_item, directory_info] = get_most_recent_item(input_path, search_folder, exclude_pattern)
% check input parameters
if nargin < 1
    error('The parent folder path is required.');
end

% set default value for search_folder
if nargin < 2 || isempty(search_folder)
    search_folder = true; 
end

% set default value for exclude_pattern
if nargin < 3 || isempty(exclude_pattern)
    exclude_pattern = ""; 
end

% initialize latest folder
most_recent_item = "";

% escape any special characters in the provided path
input_path = regexprep(input_path,'([[\]{}()=''.(),;:%%{%}!@])','\\$1');

% check input path is valid and a directory
if exist(input_path, 'dir') ~= 7
    error('The provided working directory is invalid or does not exist');
end

% get the list of all files and folders in the working directory
directory_info = dir(input_path);

if search_folder

    % filter out non-directory entries
    directory_info = directory_info([directory_info.isdir]);
    
    % remove '.' and '..' entries
    directory_info = directory_info(~ismember({directory_info.name}, {'.', '..'}));
else
    
    % filter out non-file entries
    directory_info = directory_info(~[directory_info.isdir]);

end

% if there are any directories left, get the latest
if ~isempty(directory_info)
    
    % get the modification dates of the directories
    dates = [directory_info.datenum];
    
    if strcmp(exclude_pattern, "")
        % find the index of the latest directory
        [~, latestIdx] = max(dates);
        
        % get the name of the latest directory
        most_recent_item = directory_info(latestIdx).name;
    else
        % sort dates in descending order and get the sorted indices
        [~, sorted_indices] = sort(dates, 'descend');
    
        % initialize the index for the sorted array
        idx = 1;
    
        % iterate over the sorted indices to find the most recent item without the pattern
        while idx <= length(sorted_indices)
            
            % extract current name
            current_item = directory_info(sorted_indices(idx)).name;
            
            % if the current item contains the undesired pattern or is not a csv file, move on
            if contains(current_item, exclude_pattern) || ~endsWith(current_item, '.csv')
                idx = idx + 1; 
            else
                most_recent_item = current_item;
                break;
            end
        end    
    end
end
end
