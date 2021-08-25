function [A, Q, dAdl, dAds, dQdl, dQds] = matern_to_state_sym(D)
% Convert matern kernel to state-space model
% with symbolic output
%
%
% Arguments:
%   D: Dimension
%
% Return:
%   A, Q: Transition A and covariance Q
%   dAdl: dA / dl
%   dAds: dA / dσ
%   dQdl: dQ / dl
%   dQds: dQ / dσ
%
% Zheng Zhao @ 2018
%
l = sym('l', 'real');
dt = sym('dt', 'real');
sigma = sym('sigma', 'real');

if D == 1
    
    lam = 1 / l;
    F = -lam;
    L = 1;
    
elseif D == 2
    
    lam = sqrt(sym(3)) / l;
    F = [0 1; -lam^2 -2*lam];
    L = [0; 1];
    
    
elseif D == 3
    
    lam = sqrt(sym(5)) / l;
    F = [0 1 0; 0 0 1; -lam^3 -3*lam^2 -3*lam];
    L = [0; 0; 1];
    
end

q = sigma^2 * (factorial(D-1)^2 / factorial(2*D-2)) * (2*lam)^(2*D-1);

A = expm(F * dt);
% matrix fraction
n   = size(F, 1);
Phi = [F L*q*L'; zeros(n, n) -F'];
AB  = expm(Phi*dt) * [zeros(n, n); eye(n)];
Q   = AB(1:n, :) / AB((n+1):(2*n), :);

A = simplify(A);
Q = simplify(Q);

dAdl = simplify(diff(A, l));
dAds = simplify(diff(A, sigma));
dQdl = simplify(diff(Q, l));
dQds = simplify(diff(Q, sigma));

% A = simplify(subs(A, [lam, sigma], [lam_num, sigma_num]));
% Q = simplify(subs(Q, [lam, sigma], [lam_num, sigma_num]));

end

