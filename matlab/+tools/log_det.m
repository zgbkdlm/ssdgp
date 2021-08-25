function [val] = log_det(K)
% Calculate a log determinant using Cholesky
% val = log |K|
%     = log |L L'|
%     = log |L| + log |L'|
%     = 2 * sum log diag(L)
% 
% Arguments:
%   K: A Hermitian matrix
%
% Zheng Zhao @ 2018
%

K = (K + K') / 2;
try
    L = chol(K, 'lower');
    val = 2 * sum(log(diag(L)));
catch ME
    val = log(det(K));
end

end

