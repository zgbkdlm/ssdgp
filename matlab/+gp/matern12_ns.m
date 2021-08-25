function val = matern12_ns(x, y, lx, ly, sigmax, sigmay, g)
% Non-stationary Matérn GP kernel with v=1/2 
% k(x, y) = σ(x)σ'(x) ℓ(x)^{1/4}*ℓ(x')^{1/4} * (2/(ℓ(x)+ℓ(x')))^{1/2} 
%           * exp(-r/sqrt(ℓ(x)+ℓ(x')/2))
%
% Reference:
%   Christopher J. Paciorek and Mark J. Schervish
%   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2157553/pdf/nihms13857.pdf
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

% As well as l and sigma and positivity
lx = g(lx(:));
ly = g(ly(:)');

sigmax = g(sigmax(:));
sigmay = g(sigmay(:)');

% Calculate distance
rxy = pdist2(x, y);

% Calculate σ(x) * σ(x') and l^{1/4}(x) * l^{1/4}(x') blah
Z = (sigmax * sigmay) .* (lx.^(0.25) * ly.^(0.25)) .* sqrt(2 ./ (lx + ly));
Q = rxy .* sqrt(2 ./ (lx + ly));

val = Z .* exp(-Q);

end

