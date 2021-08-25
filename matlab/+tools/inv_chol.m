function val = inv_chol(A, B)
% Efficient solving xA=B using cholesky decomp
% val = A * B^{-1}
% Let LL^T = B
% then A * B^{-1} = A / L' / L
%
% Input:
%   A, B: matrix or vector

try
    L = chol(B, 'lower');
    val = A / L' / L;
catch ME
    warning('chol failed. Use old solver instead.');
    val = A / B;
end

end

