% Composite sinusoidal regression by using a SS-DGP of the form
%
% f ~ GP(0, C(t, t'; \ell(t))),
% u21 ~ GP(0, C_{Mat.}(t, t')),
% where \ell(t) = g(u21(t)).
%
% Solved using an cubature Kalman filter and smoother.

%% Simulate data
clear
clc
close all

rng('default');
rng(2020);

f_func = @toymodels.sin_awesome;

T = [0.0005:0.0005:1];

R = 0.01 * eye(length(T)); 

y = f_func(T) + (sqrt(R) * randn(length(T), 1))';

query = T';

%% SS-DGP Construction
g = "exp";

f = dgp.DGPNode('f', [], g, 'f', @gp.matern32_ns, [1 0.38]);

u21 = dgp.DGPNode(f, 'l', g, 'u21', @gp.matern12, [2.83 1.49]); 

my_dgp = dgp.DGP(f);
my_dgp.compile()
ss = ssdgp.SSDGP(my_dgp, "TME-2");

my_dgp.load_data(T, y, R);

%% CKFS
[x_post, post, time] = filters.SGP_CKFS(ss, query, 1);
rmse = tools.RMSE(post{3}(1, :), f_func(x_post));