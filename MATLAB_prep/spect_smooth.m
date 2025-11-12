function [ spect_smooth ] = spect_smooth( spect, target_idx )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
N = length(spect);
spect_smooth = spect;
f = (1:N);

%%
local_mask = false(N, 1);
local_mask(target_idx-5:target_idx-1)=true;
local_mask(target_idx+1:target_idx+5)=true;
target_mask = false(N, 1);
target_mask(target_idx-1:target_idx+1)=true;

target_real = interp1(f(local_mask), real(spect(local_mask)), f(target_mask), 'spline');
target_imag = interp1(f(local_mask), imag(spect(local_mask)), f(target_mask), 'spline');

spect_smooth(target_mask) = target_real + 1i*target_imag;

target_idx = N + 2 - target_idx;
target_mask = false(N, 1);
target_mask(target_idx-1:target_idx+1)=true;

spect_smooth(target_mask) = target_real(end:-1:1) - 1i*target_imag(end:-1:1);

end

