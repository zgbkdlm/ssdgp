function [dkdl, dkds] = grad_matern52(x, y, l, sigma, g, dg)
% Return the numeric derivative of kernel function w.r.t. l and sigma
% evaluated at input x and y
% ∂k(*,*)/∂l | x, y
% ∂k(*,*)/∂σ | x, y
%
% Matern v=5/2
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

% https://www.wolframalpha.com/input/?i=derivative+of+a%5E2+*+%281+%2B+sqrt%285%29+*+r+%2F+g%28l%29+%2B+5+*+r%5E2+%2F+%283*g%28l%29%5E2%29%29+*+exp%28-sqrt%285%29+*+r+%2F+g%28l%29%29+with+respect+to+l

% Ensure x and y are column vector to use pdist2
if isrow(x)
    x = x';
end

if isrow(y)
    y = y';
end

% Calculate distance
r = pdist2(x, y);

dkdl = 5 * sigma^2 * r.^2 .* exp(-sqrt(5)*r/g(l)) .* (g(l) + sqrt(5)*r) * dg(l) / (3*g(l)^4);
dkds = 2 * sigma * (1 + sqrt(5) * r / g(l) + 5 * r.^2 / (3*g(l)^2)) .* exp(-sqrt(5) * r / g(l));

end

