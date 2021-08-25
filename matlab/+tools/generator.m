function [out] = generator(x, phi, f, L, Qw)
% Infinitetestomial generator
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
out = jacobian(phi, x) * f + 1/2 * trace(hessian(phi, x)' * (L*Qw*L'));

end