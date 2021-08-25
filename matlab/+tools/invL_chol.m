function val = invL_chol(A, B)
% Efficient solving Ax=B using cholesky decomp
% val = A^{-1} * B
% Let LL^T = B
% then A^{-1} * B = L' \ L \ B
%
% Input:
%   A, B: matrix or vector

try
    L = chol(A, 'lower');
    val = L' \ (L \ B);
catch ME
    warning('chol failed. Use old solver instead.');
    val = A \ B;
end

end

