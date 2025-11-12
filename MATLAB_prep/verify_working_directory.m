function working_directory = verify_working_directory(working_directory)
    % escape any special characters in the provided path
    working_directory = regexprep(working_directory,'([[\]{}()=''.(),;:%%{%}!@])','\\$1');
    
    % check path is valid and a directory
    if exist(working_directory, 'dir') ~= 7
        error('The provided working directory is invalid or does not exist');
    else
        fprintf('Working directory: %s\n', working_directory);
    end
end