function [dKdl, dKds] = grad_matern52_ns(x, l, sigma, i, g, dg)
% Return the numeric derivative of covariance w.r.t. l and sigma
% evaluated at input x and y
% Currently only support a symmetric condition k(x, x')
%
% K(l1,l1) K(l1,l2) K(l1,l3) ...
% K(l2,l1) K(l2,l2) K(l2,l3) ...
% K(l3,l1) K(l3,l2) K(l3,l3) ...
% ...
%
% ∂K(*,*)/∂l_i 
% ∂k(*,*)/∂σ_i 
%
% Matern v=5/2 
%
% Arguments:
%   x:          Evaluation points
%   l, sigma:   l(1:N) and sigma(1:N).
%   i:          l_i.  A number from 1 to N
%   g:          The function g(l), for example "exp" or "square".
%   dg:         g'(l)
%
% Return:
%   dKdl:     numerical dk(x,y)/dl_i
%   dKds:     numerical dk(x,y)/ds_i
%
% Zheng Zhao @ 2018
%

% https://www.wolframalpha.com/input/?i=derivative+of+a%5E2*exp%28-r%2Fg%28l%29%29with+respect+to+l

N = length(x);

% Ensure x and y are column vector to use pdist2
x = x(:);

% dg evaluatons
dg_li = dg(l(i));
dg_si = dg(sigma(i));

% As well as l and sigma and positivity
l = g(l(:));
sigma = g(sigma(:));

% Initialize solution matrices
dKdl = zeros(N, N);
dKds = dKdl;

% Calculate distance
r = abs(x - x(i));

% some calculations
% lx_plus_ly = l + l(i);

% Q = r .* sqrt(2 ./ lx_plus_ly);

% Deal with l_i
% From Matlab symbolic result
dKdl(:, i) = useless.matern52_ns_grad_l(sigma, sigma(i), l, l(i), r) * dg_li;
dKdl(i, :) = dKdl(:, i)';
dKdl(i, i) = useless.matern52_ns_grad_ll(sigma(i), l(i), r(i)) * dg_li;

% Deal with sigma_i
dKds(:, i) = useless.matern52_ns_grad_s(sigma, sigma(i), l, l(i), r) * dg_si;
dKds(i, :) = dKds(:, i)';
dKds(i, i) = useless.matern52_ns_grad_ss(sigma(i), sigma(i), l(i), l(i), r(i)) * dg_si;
end

