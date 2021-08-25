function val = matern52(x, y, l, sigma, g)
% Matérn GP kernel with v=5/2
% k(x, y) = ...
%
% Arguments:
%   x, y:       vector or scalar input
%   sigma, l:   hyperparameters
%   g:          additonal function on l to ensure positivity g(l), such as exp
%               or square
%
% Zheng Zhao (c) 2019
% zz@zabemon.com
%

% Ensure x and y are column vector to use pdist2
x = x(:);
y = y(:);

% Calculate distance
r = pdist2(x, y);

val = g(sigma)^2 * (1 + sqrt(5) * r / g(l) + 5 * r.^2 / (3*g(l)^2)) ...
    .* exp(-sqrt(5) * r / g(l));

end

