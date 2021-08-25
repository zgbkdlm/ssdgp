function val = weighted_sum(w, x)
%Function for calculating weight summation
% That is, val = \sum_i w_i*x_i
%
% Zheng Zhao @ 2019 Aalto University
% zz@zabemon.com 
%
% Copyright (c) 2018 Zheng Zhao
% 
% Verson 0.1, Dec 2018

% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or 
% any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.
%
    % val = sum(bsxfun(@times, w_m, sigp), 2); 
    if ndims(x) > 2
        val = zeros(size(x, 1), size(x, 2));
        for i=1:size(w, 2)  % Note that the w is a row vector
            val = val + w(i) * x(:, :, i);
        end
    else
        val = zeros(size(x, 1), 1);
        for i=1:size(w, 2)
            val = val + w(i) * x(:, i);
        end
    end
end