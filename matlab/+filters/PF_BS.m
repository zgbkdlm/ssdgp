function [x_post, post, time] = PF_BS(ss, query, to_ss, num_bs, num_par_lp, num_cores)
% Bootstrap Particle filter and backward simulation smoother
%
%
% x_k = F(x_{k-1}) + q_{k-1},
% y_k = H x_k + r_k.
%
% Arguments:
%   SS:         The SSDGP object, with loaded data in DGP.
%   query:      The query/interpolation/integration positions
%   to_ss:      Push results to ss object?
%   num_bs:     Number of (sub) backward simulation
%   num_par_lp: Number of sequencial parallel tasks
%   num_core:   Number of MATLAB workers, leave empty to use system default
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

num_samples = 200000;
dim_x = length(m0);

samples_history = zeros(dim_x, num_samples, N); 

samples = tools.gauss_rnd(m0, P0, num_samples);

for k = 1:N
    k
    for i = 1:num_samples
        samples(:, i) = F(dt(k), samples(:, i)) ...
                        + tools.chol_LDL(Q(dt(k), samples(:, i))) * randn(dim_x, 1);
    end
        
    if ~isnan(Y(:, k))
        W = exp(-0.5 * (Y(k) - samples(1, :))'.^2 / R);

        W = W / sum(W);

        % Resample
        ri = tools.resampstr(W);

        samples = samples(:, ri);
    end
    
    samples_history(:, :, k) = samples;
    m = mean(samples, 2);
    
    MM(:, k) = m;
    PP(:, :, k) = cov((samples)');
    
end

%% Backward simulation Smoother
SM_SSX = zeros(dim_x, round(num_bs * num_par_lp), N);

% Using parfeval
% On Triton with 12 worker, 100 num_bs, EM, 3 dim costs 20 min, 12 cores
% and 
%
if nargin < 4
    num_cores = 100;
end

try
    parpool(num_cores);
catch ME
    ;
end

% This is a safe way to make sure no memory leakage by packaging in a
% function. 
tic
for i = 1:num_par_lp
    fprintf('Top iteration %d/%d \n', i, num_par_lp);
    SM_SSX(:, num_bs*(i-1)+1:num_bs*i, :) = ...
                        filters.BS_PAR(samples_history, num_bs, F, Q, dt);
end
time = toc;

% Shut down parallel pool

post = {MM, PP, SM_SSX};

if to_ss
    ss.x_post = x_post;
    ss.MM = MM;
    ss.PP = PP;
    ss.MS = MS;
    ss.PS = PS;
end

    
end
