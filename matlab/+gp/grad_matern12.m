function [dkdl, dkds] = grad_matern12(x, y, l, sigma, g, dg)
% Return the numeric derivative of kernel function w.r.t. l and sigma
% evaluated at input x and y
% ∂k(*,*)/∂l | x, y
% ∂k(*,*)/∂σ | x, y
%
% Matern v=1/2 (OU)
%
% Arguments:
%   x, y:       Covariance
%   l, sigma:   hyper-parameters.
%   g:          The function g(l), for example "exp" or "square".
%   dg:         g'(l)
%
% Return:
%   dkdl:     numerical dk(x,y)/dl
%   dkds:     numerical dk(x,y)/ds
%
% Zheng Zhao @ 2018
%

% https://www.wolframalpha.com/input/?i=derivative+of+a%5E2*exp%28-r%2Fg%28l%29%29with+respect+to+l

% Ensure x and y are column vector to use pdist2
if isrow(x)
    x = x';
end

if isrow(y)
    y = y';
end

% Calculate distance
r = pdist2(x, y);

dkdl = sigma^2 * r .* exp(-r / g(l)) * dg(l) / g(l)^2;
dkds = 2 * sigma * exp(-r / g(l));

end

