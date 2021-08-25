% Test gradients of covariance functions
clear
clc

% It's very numerically weird that if precision goes to very small e.g.,
% 1e-10 ~ 1e-15, the results would be awfully different, even worse for
% smoother Matern functions.
precision = 1e-8;

x = [0.1:0.1:1]';
f = randn(length(x), 1);

l = abs(randn(10, 1));
l2 = l;
sigma = abs(randn(10, 1));
sigma2 = sigma;

i = randi(length(x));
l2(i) = l2(i) + precision;

g = @(x) exp(x);
dg = @(x) exp(x);

gp_ker = @gp.matern32_ns;

C1 = gp_ker(x, x', l, l', sigma, sigma', g);
C2 = gp_ker(x, x', l2, l2', sigma2, sigma2', g);
dKdl_fd = (C2 - C1) / precision;

obj_val1 = f' / C1 * f + tools.log_det(2*pi*C1);
obj_val2 = f' / C2 * f + tools.log_det(2*pi*C2);

[dKdl, dKds] = gp.grad_matern32_ns(x, l, sigma, i, g, dg);
alp = tools.invL_chol(C1, f);
grad_l = trace(tools.invL_chol(C1, dKdl) - alp * alp' * dKdl);

grad_l_fd = (obj_val2 - obj_val1) / precision;

