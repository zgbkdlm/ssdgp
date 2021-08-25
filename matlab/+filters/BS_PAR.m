function [SM_SSX] = BS_PAR(samples_history, num_bs, F, Q, dt)
% Backward simulation particle smoother using Matlab parallel toolbox
%
% Arguments:
%   samples_history:    Filtering particles
%   num_bs:             Number of backward simulation (efficient to be 
%                       propotional to the number of workers)
%   F, Q, dt:           Dynamic coefficients
%
% Return:
%   SM_SSX:             Smoothing trjectories
%
% Zheng Zhao 2020
%

SM_SSX = zeros(size(samples_history, 1), num_bs, size(samples_history, 3));

SM_SSX_BOX(1:num_bs) = parallel.FevalFuture;

% Send to par
for i = 1:num_bs
    SM_SSX_BOX(i) = parfeval(@filters.PF_BS_PAR_HANDLE, ...
                                1, samples_history, F, Q, dt);
end

% Fetch
for i = 1:num_bs
    [idx, val] = fetchNext(SM_SSX_BOX);
    SM_SSX(:, idx, :) = val;
    fprintf('Sub traj %d/%d finished. \n', idx, num_bs);
end

end

