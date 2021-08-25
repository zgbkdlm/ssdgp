function [x_post, post, time] = PF_BOOTSTRAP(ss, query, to_ss, num_samples)
% Bootstrap Particle filter and smoother
% x_k = F(x_{k-1}) + q_{k-1},
% y_k = H x_k + r_k.
%
% Arguments:
%   SS:         The SSDGP object, with loaded data in DGP.
%   query:      The query/interpolation/integration positions
%   to_ss:      Push results to ss object?
%   num_samples:Number of samples
%
% Returns:
%   x_post:     The location where posterior is evaluated
%   post:       The posterior estimates in a cell. {MM, PP, MS, PS}
%   time:       The CPU time {time_f, time_s} for filter and smoother
%
%
% Zheng Zhao 趙正 (c) 2020
% zz@zabemon.com
%

%% 
% Check if SS is legit
if ~isa(ss, 'ssdgp.SSDGP')
    error('SS is not a ssdgp.SSDGP class');
end

% Check if everything is okay
if ~ss.dgp.compiled
    error('DGP is not compiled.');
end

if ~ss.dgp.data_loaded
    error('Data is not load. Pls use dgp.load_data method.');
end

%% Initialize
% Here we will try to avoid directly operating on the object properties for
% performance issue, though it might be memory-inefficient

m0 = ss.m0;
P0 = ss.P0;

% Model
F = ss.F;
Q = ss.Q;
AX = ss.D;
H = zeros(1, length(m0));
H(1) = 1;

%% Data
t = ss.dgp.x;
y = ss.dgp.y;
R = ss.dgp.R(1, 1);

% Prepare data in temporal order
[T, Y, ~] = tools.interp_x(t, y, query);
x_post = T;
Y = Y';              % In filtering/smoothing, we use column of Y as time step.
dt = diff(T);
dt = [dt(1); dt];

N = length(T);

MM = zeros(length(m0), N);
PP = zeros(length(m0), length(m0), N);

%% Bootstrap particle filter

if nargin < 4
    num_samples = 200000;
end

dim_x = length(m0);

samples = tools.gauss_rnd(m0, P0, num_samples);
samples_of_f = zeros(N, num_samples); % For NLPD
Ws = zeros(N, num_samples);

tic
for k = 1:N
    k
    for i = 1:num_samples
        samples(:, i) = F(dt(k), samples(:, i)) ...
                        + tools.chol_LDL(Q(dt(k), samples(:, i))) * randn(dim_x, 1);
    end
    if ~isnan(Y(:, k))
        W = exp(-0.5 * (Y(k) - samples(1, :))'.^2 / R);

        W = W / sum(W);
        
        samples_of_f(k, :) = samples(1, :);
        Ws(k, :) = W;

        % Resample
        ri = tools.resampstr(W);

        samples = samples(:, ri);
    end
    
    m = mean(samples, 2);
    
    MM(:, k) = m;
    PP(:, :, k) = cov((samples)');
    
end
time = toc;

% Calculate NLPD without having to save samples.
f_func = @toymodels.rect;
y = f_func(ss.dgp.x) + (sqrt(ss.dgp.R) * randn(length(ss.dgp.x), 1))';

post = {MM, PP};

end
