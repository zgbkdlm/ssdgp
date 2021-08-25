function smooth_traj = PF_BS_PAR_HANDLE(samples_history, F, Q, dt)
% The function handle for backward simulation particle smoother
% used for parallezation, e.g., parfeval().
% Be careful about the memory usage though, BOOOM!
%
% Zheng Zhao 2020
%

smooth_traj = zeros(size(samples_history, 1), ...
                    size(samples_history, 3));

% Randmly choose one from end
ind = floor(rand * size(samples_history, 2) + 1);
xn = samples_history(:, ind, end);
smooth_traj(:, end) = xn;

for k = size(samples_history, 3)-1:-1:1
    
    SX = samples_history(:, :, k);

    mu = F(dt(k+1), SX);

    for j = 1:size(samples_history, 2)
        % Here is a patch. I used LDL to fix the pd problem of TME.
        Qs = tools.chol_LDL(Q(dt(k+1), SX(:, j)));
        W(j) = mvnpdf(xn, mu(:, j), Qs * Qs');
%         W(j) = mvnpdf(xn, mu(:, j), Q(dt(k+1), SX(:, j)));
    end

    W  = W ./ sum(W);

    ind = tools.categ_rnd(W);

    xn = samples_history(:, ind, k);

    smooth_traj(:, k) = xn;
    
end

end