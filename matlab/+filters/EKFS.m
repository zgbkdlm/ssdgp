function [x_post, post, time] = EKFS(ss, query, to_ss)
% Extended Kalman filter and smoother on a discretised model
% x_k = (x_{k-1}) + q_{k-1},
% y_k = H x_k + r_k.
%
% Arguments:
%   SS:         The SSDGP object, with loaded data in DGP.
%   query:      The query/interpolation/integration positions
%   to_ss:      Push results to ss object?
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
f = ss.F;
F = ss.dFdu;
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

MM_pred = MM;
PP_pred = PP;

dim_x = length(m0);

%% Filtering pass
m = m0;
P = P0;

tic
for k = 1:N
    
    % Prediction step
    P = F(dt(k), m) * P * F(dt(k), m)' + Q(dt(k), m);
    m = f(dt(k), m);
    
    MM_pred(:, k) = m;
    PP_pred(:, :, k) = P;

    % Update step
    if ~isnan(Y(:, k))
        S = H * P * H' + R;
        K = P * H' / S;
        m = m + K * (Y(:, k) - H * m);
        P = P - K * S * K';
    end
    
    MM(:, k) = m;
    PP(:, :, k) = P;
end
time_f = toc;

%% Smoothing pass

MS = MM;
PS = PP;

tic
for k = N-1:-1:1
    
    D = PP(:, :, k) * F(dt(k), MM(:, k))';
    G = D / PP_pred(:, :, k+1);
    MS(:, k) = MM(:, k) + G * (MS(:, k+1) - MM_pred(:, k+1));
    PS(:, :, k) = PP(:, :, k) + G * (PS(:, :, k+1) - PP_pred(:, :, k+1)) * G';

end
time_s = toc;

post = {MM, PP, MS, PS};
time = {time_f, time_s};

if to_ss
    ss.x_post = x_post;
    ss.MM = MM;
    ss.PP = PP;
    ss.MS = MS;
    ss.PS = PS;
end
    
end
