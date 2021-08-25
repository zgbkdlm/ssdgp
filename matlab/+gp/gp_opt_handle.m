function [obj_val, grad] = gp_opt_handle(gp_ker, XX, YY, hyper_para, R, Ky_para)
% A matlab function handle that returns the objective function and gradient
% of GP optimization. Used only for fminunc()
%
% Arguments:
%   gp_ker:     the function handle of GP kernel. For example, @gp.matern32
%   XX, YY:     data pairs of measurements
%   hyper_para: [l, sigma]
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
% Zheng Zhao @ 2018
%
l = hyper_para(1);
sigma = hyper_para(2);

g = Ky_para{1};
dg = Ky_para{2};

% Ensure YY is column vector
if isrow(YY)
    YY = YY';
end
    
Kxx = gp_ker(XX, XX, l, sigma, g);
Ky = Kxx + R;

% (neg) Log-likelihood (minimize)
obj_val = 0.5 * tools.inv_chol(YY', Ky) * YY + 0.5 * tools.log_det(2*pi*Ky);

% Calculate gradients of dKy / dl and dKy / dsigma
ker_info = functions(gp_ker);
    
if strcmp(ker_info.function, 'gp.matern12')
    
    [dKydl, dKyds] = gp.grad_matern12(XX, XX, l, sigma, g, dg);
    
elseif strcmp(ker_info.function, 'gp.matern32')
    
    [dKydl, dKyds] = gp.grad_matern32(XX, XX, l, sigma, g, dg);
    
elseif strcmp(ker_info.function, 'gp.matern52')
    
    [dKydl, dKyds] = gp.grad_matern52(XX, XX, l, sigma, g, dg);
    
else
    
    error('Wrong kernel selection');
    
end

alp = tools.invL_chol(Ky, YY);

% Give gradients [grad_l, grad_sig]
grad = -1*[0.5 * trace(alp * alp' * dKydl - tools.invL_chol(Ky, dKydl)), ...
           0.5 * trace(alp * alp' * dKyds - tools.invL_chol(Ky, dKyds))];

end

