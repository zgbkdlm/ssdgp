% Test gradients of covariance functions
clear
clc

x = [0.1:0.1:1]';

l = abs(randn(10, 1));
l2 = l;
sigma = abs(randn(10, 1));
sigma2 = sigma;

i = randi(length(x));
l2(i) = l2(i) + 1e-10;

g = @(x) exp(x);
dg = @(x) exp(x);

gp_ker = @gp.matern32_ns;

[dKdl, dKds] = gp.grad_matern32_ns(x, l, sigma, i, g, dg);

K = gp_ker(x, x', l, l', sigma, sigma', g);
K2 = gp_ker(x, x', l2, l2', sigma2, sigma2', g);

dKdl_fd = (K2 - K) / 1e-10;

