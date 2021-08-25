function val = matern12(x, y, l, sigma, g)
% Matérn GP kernel with v=1/2 (equivalent to Ornstein-Uhlenbeck)
% k(x, y) = sigma^2 * exp (-|x-y|/l)
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

val = g(sigma)^2 * exp(-r / g(l));

end

