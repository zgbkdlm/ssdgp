% test f.grad_log_pdf_ls

clear
close all

f_func = @toymodels.rect;
T = [0:0.01:1];
R = 0.01 * eye(length(T)); 
y = f_func(T) + (sqrt(R) * randn(length(T), 1))';

query = [0:0.01:1];

%% DGP Construction
g = "exp";

hyper_para = [ones(length(T), 1) ones(length(T), 1)];
hyper_last_layer = [1e-3, 1];

f = dgp.DGPNode('f', [], g, 'f', @gp.matern32_ns, hyper_para);

u21 = dgp.DGPNode(f, 'l', g, 'u21', @gp.matern12_ns, hyper_para);
u22 = dgp.DGPNode(f, 'sigma', g, 'u22', @gp.matern12_ns, hyper_para);

my_dgp = dgp.DGP(f);
my_dgp.compile()

my_dgp.load_data(T, y, R);

my_dgp.assign_u(my_dgp.U);

%%

which_i = randi(99) + 1;

obj_val = f.log_pdf();
grad_ls = f.grad_log_pdf_ls;
grad_ls = grad_ls(which_i, 1);
grad_ls

%
% It's very weird that the finite difference can not to be very small. 
f.descendants.l.u(which_i) = f.descendants.l.u(which_i) + 1e-6;

obj_val2 = f.log_pdf();

grad_ls_fd = (obj_val2 - obj_val) / 1e-6;
grad_ls_fd