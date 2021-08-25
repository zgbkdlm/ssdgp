function [x_post, post, time] = SGP_UKFS(ss, query, to_ss)
% Generic cubature filter and smoother on a discretised model
% x_k = F(x_{k-1}) + q_{k-1},
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

MM_pred = MM;
PP_pred = PP;

%% Initialize cubature sigma points
dim_x = length(m0);

alpha = 1;
beta = 0;
kappa = 3 - dim_x;
[WM, WC, c] = quadratures.ut_weights(dim_x, alpha, beta, kappa);
WM = WM';
WC = WC';

XI = [zeros(dim_x,1) eye(dim_x) -eye(dim_x)];
XI = sqrt(c) * XI;

%% Filtering pass
m = m0;
P = P0;

sigp = chol(P)' * XI + repmat(m, 1, 2*dim_x+1);
ax = sigp;
Bx = zeros(dim_x, dim_x, 2*dim_x+1);

tic
for k = 1:N
    
    % Prediction step
    sigp = chol(P)' * XI + repmat(m, 1, 2*dim_x+1);
    for j = 1:2*dim_x+1
        ax(:, j) = F(dt(k), sigp(:, j));
        Bx(:, :, j) = Q(dt(k), sigp(:, j)) + ax(:, j) * ax(:, j)';
    end
    m = quadratures.weighted_sum(WM, ax);
    P = quadratures.weighted_sum(WM, Bx) - m * m';
    
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

    sigp = chol(PP(:, :, k))' * XI + repmat(MM(:, k), 1, 2*dim_x+1);   
    % Calculate cross-cov D
    for j=1:2*dim_x+1
        Bx(:, :, j) = AX(dt(k+1), sigp(:, j));
    end
    D = quadratures.weighted_sum(WM, Bx) - MM(:, k) * MM_pred(:, k+1)';
    % Smoothing Gain
    G = D / PP_pred(:, :, k+1);
    % Smooth
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
