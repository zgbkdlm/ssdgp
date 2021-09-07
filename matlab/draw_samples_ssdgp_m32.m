%% Draw a few DGP samples from an SS-DGP model
% f ~ GP(; U1, U2)
% U1 ~ GP Matern 3/2
% U2 ~ GP Matern 3/2
%

%%
clear
clc
close all
rng('default')
rng(999)

%% Time
dt = 0.001;
t = [dt:dt:10];

%% Build model
g = 'exp';

f = dgp.DGPNode('f', [], g, 'f', @gp.matern32_ns, [1, 1]);

u21 = dgp.DGPNode(f, 'l', g, 'u21', @gp.matern32, [1, 1]); 
u22 = dgp.DGPNode(f, 'sigma', g, 'u22', @gp.matern32, [1, 1]);

my_dgp = dgp.DGP(f);
my_dgp.compile()
ss = ssdgp.SSDGP(my_dgp, "TME-3");

a = ss.F;
Q = ss.Q;
dim = ss.dim;

m0 = zeros(dim, 1);
P0 = 2 * eye(dim);

%% Simulate SDE with Gaussian increments
num_traj = 20;
int_steps = 10;
ddt = dt / int_steps;

uu = zeros(num_traj, dim, length(t));

for n = 1:num_traj
    u0 = m0 + chol(P0, 'lower') * randn(dim, 1);
    u = u0;
    for i = 1:length(t)
        for j = 1:int_steps
            u = a(ddt, u) + chol(Q(ddt, u), 'lower') * randn(dim, 1);
        end
        uu(n, :, i) = u;
    end
end

%% Plot and save figs
mkdir figs
plot(t, squeeze(uu(1, 1, :)), 'Color', 'k', 'LineWidth', 2)
xlabel('$t$', 'Interpreter', 'latex')
ylabel('$f$', 'Interpreter', 'latex')
title("Samples drawn from a Mat\'{e}rn 3/2 SS-DGP", 'Interpreter', 'latex');
xlim([0, 10]);
lgd = legend('Location', 'northwest');
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
    
for i = 1:num_traj
    name = sprintf('Sample %d', i);
    plot(t, squeeze(uu(i, 1, :)), 'Color', 'k', 'LineWidth', 2, 'Displayname', name)

    xlabel('$t$', 'Interpreter', 'latex')
    ylabel('$u(t)$', 'Interpreter', 'latex')
    title("Samples drawn from a Mat\'{e}rn 3/2 SS-DGP $U(t)$", 'Interpreter', 'latex');
    xlim([0, 10]);
    lgd = legend('Location', 'northwest');
    lgd.FontSize = 18;
    lgd.Interpreter = 'latex';

    set(gca,'TickLabelInterpreter','latex')
    grid on
    ax = gca;
    ax.Box = 'off';
    ax.GridLineStyle = '--';
    ax.GridAlpha = 0.2;
    set(ax, 'FontSize', 16);
    
    file_name = sprintf('figs/sample_%d.png', i);
    print(file_name, '-dpng')
    
    pause(2)
end

% Then use ffmpeg to generate gif from the pngs above.

