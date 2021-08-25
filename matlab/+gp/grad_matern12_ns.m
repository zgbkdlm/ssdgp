function [dKdl, dKds] = grad_matern12_ns(x, l, sigma, i, g, dg)
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
% Matern v=1/2 (OU)
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
lx_plus_ly = l + l(i);

Q = r ./ sqrt(lx_plus_ly / 2);

% Deal with l_i
% From Matlab symbolic result
dKdl(:, i) = useless.matern12_ns_grad_l(sigma, sigma(i), l, l(i), r) * dg_li;
dKdl(i, :) = dKdl(:, i)';
dKdl(i, i) = sigma(i)^2 * r(i) * exp(-r(i)/sqrt(l(i))) / (2 * l(i)^(1.5)) * dg_li;

% Deal with sigma_i
% https://www.wolframalpha.com/input/?i=derivative+of+g1*g2*g3%5E0.25*g4%5E0.25*%282%2F%28g3%2Bg4%29%29%5E0.5*exp%28-r*sqrt%282%2F%28g3%2Bg4%29%29%29+with+respect+to+g2
dKds(:, i) = sigma .* l.^0.25 * l(i)^0.25 .* sqrt(2./lx_plus_ly) .* exp(-Q) * dg_si;
dKds(i, :) = dKds(:, i)';
dKds(i, i) = 2 * sigma(i) * exp(-(r(i)/sqrt(l(i)))) * dg_si;

end

