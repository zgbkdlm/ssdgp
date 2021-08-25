function [A, Q] = matern_to_state(D, l, sigma, dt)
% Convert matern kernel to state-space model
% with symbolic output
%
%
% Arguments:
%   D:          Dimension
%   l, sigma:   Parameters
%   dt:         Time interval (symbolic)
%
% Return:
%   A, Q: Transition A and covariance Q
%
% Zheng Zhao @ 2018
% Simo Sarkka
%

% Force symbolic
lam = sym('lam', 'real');
l = sym(l);
sigma = sym(sigma);
D = sym(D);

if D == 1
    
    F = -lam;
    L = 1;
    
elseif D == 2
    
    F = [0 1; 
        -lam^2 -2*lam];
    L = [0; 1];
    
    
elseif D == 3
    
    F = [0 1 0; 
        0 0 1; 
        -lam^sym(3) -3*lam^2 -3*lam];
    L = [0; 0; 1];
    
end

q = sigma^2 * (factorial(D-1)^2 / factorial(2*D-2)) * (2*lam)^(2*D-1);

A = expm(F * dt);
% matrix fraction
n   = size(F, 1);
Phi = [F L*q*L'; zeros(n, n) -F'];
AB  = expm(Phi*dt) * [zeros(n, n); eye(n)];
Q   = AB(1:n, :) / AB((n+1):(2*n), :);

if D == 1
    A = simplify(subs(A, lam, 1/l));
    Q = simplify(subs(Q, lam, 1/l));
elseif D == 2
    A = simplify(subs(A, lam, sqrt(sym(3))/l));
    Q = simplify(subs(Q, lam, sqrt(sym(3))/l));
elseif D == 3
    A = simplify(subs(A, lam, sqrt(sym(5))/l));
    Q = simplify(subs(Q, lam, sqrt(sym(5))/l));
end

end

