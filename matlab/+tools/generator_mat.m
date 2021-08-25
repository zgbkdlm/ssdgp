function b = generator_mat(x, phi, f, L, Qw)
% Generator applying on a matrix
%
% Inputs:
%   See TME()
%
% References:
%
%     [1] Zheng Zhao, Toni Karvonen, Roland Hostettler, Simo Särkkä, 
%         Taylor Moment Expansion for Continuous-discrete Filtering. 
%         IEEE Transactions on Automatic Control. 
%
% Zheng Zhao @ 2019 Aalto University
% zz@zabemon.com 
%
% Copyright (c) 2018 Zheng Zhao
% 
% Version 0.2, Mar 2021 
%   - Optmized the code by directly applying Theorem 1 for faster speed.
%   - Changed notation Q -> Qw.
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
sizes = size(phi);

b = sym(zeros(sizes));

for i = 1:sizes(1)
    for j = 1:sizes(2)
        b(i, j) = tools.generator(x, phi(i, j), f, L, Qw);
    end
end
    
end