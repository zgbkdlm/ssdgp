function [dKxxdl, dKxxds] = grad_fast_matern12(Kxx, r, l, sigma, l_func)
% Return the numeric derivative of kernel function w.r.t. l and sigma
% evaluated at input x and y BY REUSING r and Kxx.
% ∂k(*,*)/∂l | x, y
% ∂k(*,*)/∂σ | x, y
%
% Matern v=1/2 (OU)
%
% Arguments:
%   Kxx:       Covariance
%   r:         distance matrix
%   l, sigma:  hyper-parameters.
%   l_func:    The function f(l), for example "exp" or "square".
%
% Return:
%   dKxxdl:   numerical dKxx/dl
%   dKxxds:   numerical dKxx/dsigma
%
% Zheng Zhao @ 2018
%

if strcmp(l_func, 'exp')
    
    dKxxdl = Kxx .* r / exp(l);
    
elseif strcmp(l_func, 'square')
    
    dKxxdl = Kxx .* r * 2 / l^3;
    
elseif strcmp(l_func, 'none')
    
    dKxxdl = Kxx .* r / l^2;
    
end

dKxxds = 2 * Kxx / sigma;

end

