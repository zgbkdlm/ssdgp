function [a, Sigma] = TME(x, f, L, Qw, dt, order, simp)
% Giving the first two moments and covariance estimates from an SDE using
% the Taylor Moment Expansion (TME) method. Please see the references below
% for details.
%
% dx = f(x, t) dt + L(x, t) dB, 
%
% Notice that the code below is currently operated element-wisely for any 
% vector input, which might be slow for some cases.
%
% The inputs except `order` and `simp` must be symbolic
%
% Input:
%     x:      State vector (column)
%     f:      Drift function
%     L:      Dispersion function
%     Qw:     Spectral density of B(t)
%     dt:     Time interval dt
%     order:  Order of expansion
%     simp:   Set "simplify" to output simplified results
% 
% Output:
%     a:      E[x(t+dt) | x(t)]
%     B:      E[x(t+dt)x^T(t+dt) | x(t)]
%     Sigma:  Cov[x(t+dt) | x(t)]
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

dim_x = size(x, 1);

%% Mean estiamte
phi = x;
a = x;
for r = 1:order
    phi = tools.generator_mat(x, phi, f, L, Qw);
    a = a + 1 / factorial(r) * phi * dt ^ r;
end

if simp
    a = simplify(a);
end
fprintf('Symbolic TME mean approximation gave. \n');

%% Covariance
% Pre-compute generator powers
Ax = tools.generator_power(x, x, f, L, Qw, order);
Axx = tools.generator_power(x, x * x', f, L, Qw, order);

Sigma = sym(zeros(dim_x, dim_x));
for r = 1:order
    coeff = Axx{r + 1};
    for s = 0:r
        coeff = coeff - nchoosek(r, s) * Ax{s + 1} * Ax{r - s + 1}';
    end
    Sigma = Sigma + 1 / factorial(r) * coeff * dt ^ r;
end

if simp
    Sigma = simplify(Sigma);
end
fprintf('Symbolic TME covariance approximation gave. \n');

end