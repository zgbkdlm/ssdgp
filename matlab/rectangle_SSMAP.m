%% Simulate data
clear
clc
close all

rng('default');
rng(2020);

f_func = @toymodels.rect;

T = [0:0.01:1];

R = 0.002 * eye(length(T)); 

y = f_func(T) + (sqrt(R) * randn(length(T), 1))';

query = T';

%% DGP Construction
g = "exp";
model = 'ls';

switch model

    case 'ls'
        
        hyper_para = [1, 5];    % the values here do not matter
        hyper_para2 = [0.008 0.14];

        f = dgp.DGPNode('f', [], g, 'f', @gp.matern32_ns, hyper_para);

        u21 = dgp.DGPNode(f, 'l', g, 'u21', @gp.matern12, hyper_para2); 
        u22 = dgp.DGPNode(f, 'sigma', g, 'u22', @gp.matern12, hyper_para2);
        
    case 'lsls'
        
        hyper_para = [1, 5];    % the values here do not matter
        hyper_para2 = [0.001 0.92];

        f = dgp.DGPNode('f', [], g, 'f', @gp.matern32_ns, hyper_para);

        u21 = dgp.DGPNode(f, 'l', g, 'u21', @gp.matern12, hyper_para2); 
        u22 = dgp.DGPNode(f, 'sigma', g, 'u22', @gp.matern12, hyper_para2);
        u31 = dgp.DGPNode(u21, 'l', g, 'u31', @gp.matern12, hyper_para2);
        u32 = dgp.DGPNode(u21, 'sigma', g, 'u32', @gp.matern12, hyper_para2);
        u33 = dgp.DGPNode(u22, 'l', g, 'u33', @gp.matern12, hyper_para2);
        u34 = dgp.DGPNode(u22, 'sigma', g, 'u34', @gp.matern12, hyper_para2);
        
end

my_dgp = dgp.DGP(f);
my_dgp.compile()
ss = ssdgp.SSDGP(my_dgp, "TME-3");
my_dgp.load_data(T, y, R);

% Start optimization
options = optimoptions('fmincon','Algorithm','Interior-Point', ...
                                    'HessianApproximation', 'lbfgs', ... 
                                    'SpecifyObjectiveGradient',true, 'Display', ...
                                    'iter-detailed', 'MaxIterations', 3000, ...
                                    'CheckGradients', false);

[neg_likelihood] = ss.MAP([-2 3], options);

rmse = tools.RMSE(ss.U(1, 2:end), f_func(T));

%% Plot f(t)
figure()
plot(T, f_func(T), '--', 'LineWidth', 2, 'Color', [0 0.4470 0.7410], ...
    'DisplayName', 'True $f(t)$');
hold on

scatter(T, y, 300, '.', 'MarkerEdgeColor', [0.8500 0.3250 0.0980], ...
    'DisplayName', 'Measurements')

prop_line = {'Color', 'k', 'LineWidth', 3, 'DisplayName', 'SS-MAP $f(t)$'};
          
plot(T, ss.U(1, 2:end), prop_line{:});

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

prop_line = {'Color', 'k', 'LineWidth', 3, 'DisplayName', 'SS-MAP $\log(\ell^2_{1,1}(t))$'};
          
plot(T, ss.U(3, 2:end), prop_line{:});

xlabel('$t$', 'Interpreter', 'latex')
ylabel('$\log(\ell^2_{1,1}(t))$', 'Interpreter', 'latex')

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

%% Plot sigma(t)
figure()

prop_line = {'Color', 'k', 'LineWidth', 3, 'DisplayName', 'SS-MAP $\log(\sigma^2_{1,2}(t))$'};
          
plot(T, ss.U(4, 2:end), prop_line{:});

xlabel('$t$', 'Interpreter', 'latex')
ylabel('$\log(\sigma^2_{1,2}(t))$', 'Interpreter', 'latex')

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
