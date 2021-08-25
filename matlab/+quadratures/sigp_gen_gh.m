function [sigp] = sigp_gen_gh(m, P, x_points)
%Generate sigma points for Gauss-Hermite integration with given m and P
%
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

sigp = x_points;

for i = 1:size(x_points, 2)
    sigp(:, i) = m + sqrt(2) * chol(P)' * x_points(:, i);
end

end

