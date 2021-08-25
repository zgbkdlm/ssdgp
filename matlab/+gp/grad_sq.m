function [dKxxdl, dKxxds, dkdl, dkds] = grad_sq(Kxx, r, l, sigma)
% Return the derivative w.r.t. l and sigma of Kxx or kernel function, i.e.,
% Squared exponential
%
% Arguments:
%   Kxx:       Covariance
%   r:         a matrix elemently-wisely contain the distance r. 
%   l, sigma:  hyper-parameters.
%
% Return:
%   dKxxdl:   numerical dKxx/dl
%   dKxxds:   numerical dKxx/dsigma
%   dkdl:     function handle dk(x,y)/dl
%   dkds:     function handle dk(x,y)/ds
%
% Zheng Zhao @ 2018
%

% https://www.wolframalpha.com/input/?i=derivative+of+a%5E2*(1%2Bsqrt(3)*r%2Fl)*exp(-sqrt(3)*r%2Fl)+with+respect+to+l
dkdl = @(x, y, l, sigma) sigma^2 * sum((x - y).^2) * exp(-sum((x - y).^2)/(2*l^2)) / l^3;
dkds = @(x, y, l, sigma) 2 * sigma * exp(-sum((x - y).^2)/(2*l^2));

% Leave Kxx, r, l, sigma to empty if you only want gradient function 
% handle of kernels
if ~isempty(Kxx) && ~isempty(r) && ~isempty(l) && ~isempty(sigma)
    dKxxdl = Kxx .* r.^2 / l^3;
    dKxxds = 2 * Kxx / sigma;
else
    error('Trying to give derivatives but inputs are not complete.')
end

end

