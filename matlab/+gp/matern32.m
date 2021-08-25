function val = matern32(x, y, l, sigma, g)
% Matérn GP kernel with v=3/2
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

val = g(sigma)^2 * (1 + sqrt(3) * r / g(l)) .* exp(-sqrt(3) * r / g(l));

end

