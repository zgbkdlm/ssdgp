% Composite sinusoidal regression by using a SS-DGP of the form
%
% f ~ GP(0, C(t, t'; \ell(t))),
% u21 ~ GP(0, C_{Mat.}(t, t')),
% where \ell(t) = g(u21(t)). 
%
% Solved using an extended Kalman filter and smoother.
% 
% This should correspond to the Fig. 12 in the paper.
%

%% Simulate data
clear
clc
close all

% Set random seed
rng('default');
rng(2020);

% Choose the test signal
f_func = @toymodels.sin_awesome;

% Time instances
T = [0.0005:0.0005:1];

% Measurement noise covariance
R = 0.01 * eye(length(T)); 

% Generate measurements
y = f_func(T) + (sqrt(R) * randn(length(T), 1))';

% Query points (you can change this to any intervals you want)
query = T';

%% SS-DGP Construction
% Transformation function be g(u) = exp(u)
g = "exp";

% The first dgp node
f = dgp.DGPNode('f', [], g, 'f', @gp.matern32_ns, [1 1.6]);

% The second dgp node that parametrises f in the ell parameter
% Please check README.md or the docstring of the class to understand what
% do these arguments mean.
u21 = dgp.DGPNode(f, 'l', g, 'u21', @gp.matern12, [0.2 1.16]); 

% Now make a DGP instance
my_dgp = dgp.DGP(f);

% Integration and error checks (not really compile).
my_dgp.compile()

% Make an SS-DGP instance based on the DGP instance
% You can also try use "local", "EM", or "TME-order"
ss = ssdgp.SSDGP(my_dgp, "TME-2");

% Load data
my_dgp.load_data(T, y, R);

%% Perform EKFS regression

% x_post means the times where you want the posterior distribution, it's
% related to you setted `query` and `T` above.
% post is a cell containing the filtering mean, cov, and smoothing mean,
% cov.
[x_post, post, time] = filters.SGP_CKFS(ss, query, 1);
rmse = tools.RMSE(post{3}(1, :), f_func(x_post));

%% Plot f(t)
figure()

plot(T, f_func(T), '--', 'LineWidth', 2, 'Color', [0 0.4470 0.7410], ...
    'DisplayName', 'True $f(t)$');
hold on

scatter(T, y, 7, '.', 'MarkerEdgeColor', [0.8500 0.3250 0.0980], ...
    'DisplayName', 'Measurements')

prop_line = {'Color', 'k', 'LineWidth', 3, 'DisplayName', 'EKFS $f(t)$'};
prop_patch = {'FaceColor', 'k', 'EdgeColor', 'none', 'FaceAlpha', 0.2, ...
              'HandleVisibility', 'off'};
          
idx = ss.node_idx('f', 1);
tools.errBar(x_post, post{3}(idx,:), 1.96*sqrt(squeeze(post{4}(idx,idx,:))), ...
                prop_line, prop_patch);

xlabel('$t$', 'Interpreter', 'latex')
ylabel('$f$', 'Interpreter', 'latex')
ylim([-0.2, 1.2])
lgd = legend('Location', 'northeast');
lgd.FontSize = 18;
lgd.Interpreter = 'latex';

set(gca,'TickLabelInterpreter','latex')
grid on
ax = gca;
ax.Box = 'off';
ax.GridLineStyle = '--';
ax.GridAlpha = 0.2;
set(ax, 'FontSize', 16);

tools.rm_white_space()

%% Plot ell(t)
figure()

prop_line = {'Color', 'k', 'LineWidth', 3, 'DisplayName', 'EKFS $\log(\ell^2_{1,1}(t))$'};
prop_patch = {'FaceColor', 'k', 'EdgeColor', 'none', 'FaceAlpha', 0.2, ...
              'HandleVisibility', 'off'};
          
idx = ss.node_idx('u21', 1);
tools.errBar(x_post, post{3}(idx,:), 1.96*sqrt(squeeze(post{4}(idx,idx,:))), ...
                prop_line, prop_patch);

xlabel('$t$', 'Interpreter', 'latex')
ylabel('$\log(\ell^2_{1,1}(t))$', 'Interpreter', 'latex')

lgd = legend('Location', 'southeast');
lgd.FontSize = 24;
lgd.Interpreter = 'latex';

set(gca,'TickLabelInterpreter','latex')
grid on
ax = gca;
ax.Box = 'off';
ax.GridLineStyle = '--';
ax.GridAlpha = 0.2;
set(ax, 'FontSize', 16);

tools.rm_white_space()