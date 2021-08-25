function [L] = chol_LDL(A)
%Cholesky decomposition done by LDL. 
% To prevent from small singular value problem or positive semi-definite
% matrix. But be careful, this LDL might fail or give un-reasonable results
% for some singular matrices.
%
% A = L * D * L'
% chol(A, 'lower') = L * sqrt(abs(D)),
% where we force D to be positive
%
% Zheng Zhao
%

[L, D] = ldl(A);
L = L * sqrt(abs(D));

end

