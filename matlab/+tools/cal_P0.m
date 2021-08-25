function [P_inf] = cal_P0(kernel, l, sigma)
% Statainary covariance P0
%

switch kernel
        
    case 'gp.ker_matern12'
        P_inf = sigma^2;
        
    case 'gp.ker_matern32'
        lam = sqrt(3)/l;
        P_inf = [sigma^2 0; 0 lam^2*sigma^2];
        
    case 'gp.ker_matern52'
        % Check matern52_stationary_cov_derivation.nb for derivation
        lam = sqrt(5)/l;
        P_inf = [sigma^2 0 -1/3*lam^2*sigma^2; 
                  0 1/3*lam^2*sigma^2 0; 
                  -1/3*lam^2*sigma^2 0 lam^4*sigma^2];
end

end

