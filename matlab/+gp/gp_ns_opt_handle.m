function [obj_val, grad] = gp_ns_opt_handle(gp_ker, XX, YY, hyper_para, R, Ky_para)
% A matlab function handle that returns the objective function and gradient
% of GP optimization. Used only for fminunc()
%
% Arguments:
%   gp_ker:     the function handle of GP kernel. For example, @gp.matern32
%   XX, YY:     data pairs of measurements
%   hyper_para: [l(1:N), sigma(1:N)]
%   epoch:      number of iteration
%   learn_rate: learning rate
%   Ky_para:    parameters of Ky function handle. A cell containing g and
%               dg
%
% Return:
%   obj_val:    The log-likelihood value
%   grad:       gradients of l and sigma in the same vector form with
%               hyper_para
%
% ver 1.1. Feb, 2020. Add support for non-stationary kernel (Zheng)
%
% Zheng Zhao @ 2018
%

XX = XX(:);
YY = YY(:);
hyper_para = hyper_para(:);

N = length(XX);

l = hyper_para(1:N);
sigma = hyper_para(N+1:end);

g = Ky_para{1};
dg = Ky_para{2};
    
Kxx = gp_ker(XX, XX', l, l', sigma, sigma', g);
Ky = Kxx + R;

% (neg) Log-likelihood (minimize)
obj_val = 0.5 * tools.inv_chol(YY', Ky) * YY + 0.5 * tools.log_det(2*pi*Ky);

grad = zeros(1, 2*N);
% Calculate gradients of dKy / dl and dKy / dsigma
ker_info = functions(gp_ker);

if strcmp(ker_info.function, 'gp.matern12_ns')
    
    grad_func = @gp.grad_matern12_ns;
    
elseif strcmp(ker_info.function, 'gp.matern32_ns')
    
    grad_func = @gp.grad_matern32_ns;
    
elseif strcmp(ker_info.function, 'gp.matern52_ns')
    
    grad_func = @gp.grad_matern52_ns;
    
else
    
    error('Wrong kernel selection');
    
end

alp = tools.invL_chol(Ky, YY);

for i = 1:N
    % Get dK/dl and dK/ds, which are Christian cross-like matrices
    [dKydl, dKyds] = grad_func(XX, l, sigma, i, g, dg);
    
    grad(i) = -1*(0.5 * trace(alp * alp' * dKydl - tools.invL_chol(Ky, dKydl)));
    grad(N+i) = -1*(0.5 * trace(alp * alp' * dKyds - tools.invL_chol(Ky, dKyds)));
end


end

