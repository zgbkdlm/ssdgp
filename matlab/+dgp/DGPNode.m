% Class for a node of Deep Gaussian processes
%
% Notes: in the code below I used father intead of child for announcing
% dependencies like (a) -> (b), where (b) is a father of (a).
%
% Zheng Zhao (c) 2019
% zz@zabemon.com
%
classdef DGPNode < handle
    %This is an abstract class for a GP that is used as a node in the
    %contruction of DGP
    
    properties
        % Properties of the node
        gp_ker
        grad_func
        stationary   % Boolean. 1 is stationary
        hyper_para
        name
        layer        % The layer of this node
        g            % g(l)
        dg           % dg(l)/dl
        
        dgp          % Object. The DGP obejct that this node belongs to
        father       % Object. The father node
        U_idx        % The unique index in dgp.U
        compiled     % Boolean. If this node belongs to a DGP and compiled
        
        % The node's descendants
        descendants
        
        % Reserved symbolic for state space method
        sym_u
        SS_idx       % The unique index in the state vector. (first comonent)
        SS_dim       % The SS dimension of this GP
        
        % Dummy variable for storing MAP/HMC estimates when training
        % can be the initial value
        u
        
        % For MAP
        m_MAP
        
        % For HMC
        m_HMC
        cov_HMC
    end
    
    methods
        function obj = DGPNode(father, role, g, name, gp_ker, hyper_para)
            % Constructor and initialize
            %
            % Arguments:
            %   father:     Father node (a DGPNode object).
            %   role:       This node is a GP prior on the "role" part of
            %               the father node. For example role="l" means 
            %               that you put a GP on the length scale.
            %               role="sigma" means the magnitude parameter.
            %   g:          The function to ensure some property of THIS 
            %               node's hyperparameters. For example g="exp" 
            %               to ensure the positivity by exp(l). 
            %   name:       A name of this node. Must be unique aross the
            %               whole DGP. The uniqueness will be checked in
            %               DGP compile.
            %   gp_ker:     GP kernel. E.g., @gp.matern32.
            %   hyper_para: Hyperparameters of GP. If this is a stationary
            %               GP or DGP, the hyperparameter should be simply 
            %               [l, sigma], otherwise should be a two column
            %               array [l(1:N), sigma(1:N)], same length with
            %               your measurements, where l and sigma are column
            %               vectors.
            %
            
            % Check if GP kernel is supported
            % Note that you can only use stationary kernel in the GPs of
            % the last layer.
            ker_info = functions(gp_ker);
            if any(strcmp({'gp.matern12_ns', 'gp.matern32_ns', 'gp.matern52_ns', ...
                           'gp.matern12', 'gp.matern32', 'gp.matern52' }, ...
                    ker_info.function))
            else
                error('Unsupported covariance function.')
            end
            
            % Initlization grad_func for covariance function
            if strcmp(ker_info.function, 'gp.matern12_ns')
                obj.grad_func = @gp.grad_matern12_ns;
            elseif strcmp(ker_info.function, 'gp.matern32_ns')
                obj.grad_func = @gp.grad_matern32_ns;

            elseif strcmp(ker_info.function, 'gp.matern52_ns')
                obj.grad_func = @gp.grad_matern52_ns;
            else
                obj.grad_func = [];
            end
            
            if contains(ker_info.function, 'ns')
                obj.stationary = 0;
            else
                obj.stationary = 1;
            end
            
            % Build connections with father
            if strcmp(father, 'f')
                % The all father node has no father
                obj.layer = 0;
            else
                % Connect with its father
                father.descendants.(role) = obj;
                obj.layer = father.layer + 1;
            end
            
            % Initialize properties
            obj.name = name;
            obj.gp_ker = gp_ker;
            if strcmp(g, 'exp')
                g = @(l) exp(l);
                obj.dg = @(l) exp(l);
                obj.g = g;
            elseif strcmp(g, 'square')
                g = @(l) l.^2 + 1e-4;
                obj.dg = @(l) 2*l;
                obj.g = g;
            elseif strcmp(g, 'exp-lin')
                g = @tools.exp_lin;
                obj.dg = @tools.exp_lin_deri;
                obj.g = g;
            elseif strcmp(g, 'none')
                g = @(l) l;
                obj.dg = @(l) 1;
                obj.g = g;
            else
                error('Unsupported function g().')
            end
            obj.hyper_para = hyper_para;
            
            % State space properties
            ker_info = functions(gp_ker);
            if contains(ker_info.function, 'matern12')
                obj.SS_dim = 1;
            elseif contains(ker_info.function, 'matern32')
                obj.SS_dim = 2;
            elseif contains(ker_info.function, 'matern52')
                obj.SS_dim = 3;
            end
            
            % Initialize descendants with numeric hyper-paras, which will
            % be changed to objects if has sons.
            obj.descendants.l = hyper_para(:, 1);
            obj.descendants.sigma = hyper_para(:, 2);
        end
        
        %% Functions for Single GP Regression 
        function [m, cov] = regression(obj, x, y, query, R)
            % Single stationary GP regression. 
            % f ~ GP(0, C(x, x'))
            % y = f(x) + r,   r ~ N(0, R)
            %
            % Arguments:
            %   x, y:   Measurement data pairs (both column vectors).
            %   query:  where you want prediction.
            %   R:      Measurement noise covariance.
            %
            % Returns:
            %   m, cov: Mean and covariance at location 'query'.
            %
            
            if ~obj.stationary
                error('The GP is non-stationary, use regression_ns instead.');
            end
            
            x = x(:);
            y = y(:);
            query = query(:);
            
            l = obj.hyper_para(:, 1);
            sigma = obj.hyper_para(:, 2);
            
            Kxx = obj.gp_ker(x, x, l, sigma, obj.g);

            Kxxs = obj.gp_ker(x, query, l, sigma, obj.g);
            Kxsx = Kxxs';

            Kxsxs = obj.gp_ker(query, query, l, sigma, obj.g);
            
            z = tools.inv_chol(Kxsx, Kxx + R);
            m = z * y;
            cov = Kxsxs - z * Kxxs;
        end
        
        function [m, cov] = regression_ns(obj, x, y, query_x, query_l, query_s, R)
            % Single non-stationary GP regression. 
            % f ~ GP(0, C(x, x';l(x),l(x'),s(x),s(x')))
            % y = f(x) + r,   r ~ N(0, R)
            %
            % Arguments:
            %   x, y:       Measurement data pairs (both column vectors).
            %   query_x:    where you want prediction.
            %   query_l,s:  The query length scale and sigma.  
            %   R:          Measurement noise covariance.
            %
            % Returns:
            %   m, cov: Mean and covariance at location 'query'.
            %
            
            if obj.stationary
                error('The GP is stationary, use regression instead.');
            end
            
            if isempty(obj.dgp)
                l = obj.hyper_para(:, 1);
                sigma = obj.hyper_para(:, 2);
            else
                l = obj.dgp.U(:, obj.descendants.l.U_idx);
                sigma = obj.dgp.U(:, obj.descendants.sigma.U_idx);
            end
            
            Kxx = obj.gp_ker(x, x, l, l', sigma, sigma', obj.g);

            Kxxs = obj.gp_ker(x, query_x, l, query_l', sigma, query_s', obj.g);
            Kxsx = Kxxs';

            Kxsxs = obj.gp_ker(query_x, query_x', query_l, query_l', query_s, query_s', obj.g);
            
            z = tools.inv_chol(Kxsx, Kxx + R);
            m = z * y;
            cov = Kxsxs - z * Kxxs;
        end
        
        function [hyper_para, neg_log_likeli] = hyper_opt_ns(obj, x, y, R, options)
            % Optimizing hyperparameters on mariginal likelihood on a
            % non-stationary GP.
            %
            % Arguments:
            %   x, y:       Measurement data pairs (both column vectors).
            %   query:      where you want prediction.
            %   R:          Measurement noise covariance.
            %   options:    fmincon options. If not given use default.
            %
            % Returns:
            %   hyper_para: [l(1:N), sigma(1:N)].
            %
            
            if obj.stationary
                error('The GP is stationary, use hyper_opt instead.');
            end
            
            opt_func = @(hyper_para) gp.gp_ns_opt_handle(obj.gp_ker, x, y, ...
                                    hyper_para, R, {obj.g, obj.dg});
            
            if nargin < 5
                options = optimoptions('fmincon','Algorithm','Interior-Point', ...
                                    'HessianApproximation', 'lbfgs', ... 
                                    'SpecifyObjectiveGradient',true, 'Display', ...
                                    'iter-detailed', 'MaxIterations', 3000, ...
                                    'CheckGradients', false);
            end
            
            % Set constrain ranges according to function g
            g_info = functions(obj.g);
            if strcmp(g_info.function, '@(l)exp(l)')
                lower_l = -8 * ones(length(y), 1);
                lower_s = -8 * ones(length(y), 1);
                upper_l = 10 * ones(length(y), 1);
                upper_s = 10 * ones(length(y), 1);
            elseif strcmp(g_info.function, '@(l)l.^2')
                lower_l = -5 * ones(length(y), 1);
                lower_s = -5 * ones(length(y), 1);
                upper_l = 5 * ones(length(y), 1);
                upper_s = 5 * ones(length(y), 1);
            elseif strcmp(g_info.function, '@tools.exp_lin')
                lower_l = -6 * ones(length(y), 1);
                lower_s = -4 * ones(length(y), 1);
                upper_l = 20 * ones(length(y), 1);
                upper_s = 20 * ones(length(y), 1);
            elseif strcmp(g_info.function, 'none')
                lower_l = 1e-3 * ones(length(y), 1);
                lower_s = 1e-2 * ones(length(y), 1);
                upper_l = 20 * ones(length(y), 1);
                upper_s = 20 * ones(length(y), 1);
            else
                error('Unsupported function g().')
            end

            [hyper_para, neg_log_likeli] = fmincon(opt_func, obj.hyper_para(:),...
                            [], [], [], [], [lower_l, lower_s], [upper_l, upper_s], [], options);
            
            hyper_para = reshape(hyper_para, [length(y), 2]);
        end
        
        function [hyper_para, neg_log_likeli] = hyper_opt(obj, x, y, R, options)
            % Optimizing hyperparameters on mariginal likelihood on a
            % stationary GP.
            %
            % Arguments:
            %   x, y:       Measurement data pairs (both column vectors).
            %   R:          Measurement noise covariance.
            %   options:    fmincon options. If not given use default.
            %
            % Returns:
            %   hyper_para: [l, sigma].
            %
            
            if ~obj.stationary
                error('The GP is non-stationary, use hyper_opt_ns instead.');
            end
            
            opt_func = @(hyper_para) gp.gp_opt_handle(obj.gp_ker, x, y, ...
                                    hyper_para, R, {obj.g, obj.dg});
            
            if nargin < 5
                options = optimoptions('fmincon','Algorithm','Interior-Point', ...
                                    'HessianApproximation', 'lbfgs', ... 
                                    'SpecifyObjectiveGradient',true, 'Display', ...
                                    'iter-detailed', 'MaxIterations', 3000, ...
                                    'CheckGradients', false);
            end

            [hyper_para, neg_log_likeli] = fmincon(opt_func, obj.hyper_para(:),...
                            [], [], [], [], [0, 0], [10, 10], [], options);
                                    
        end
        
        %% Functions for Involving in DGP 
        function val = log_pdf(obj, x, u)
            % Give the evaluation of log probability density at arbitary 
            %   point u(x_{1:N}). It will use the descendant information 
            %   to obtain the hyperpameter values if needed.
            %
            % val = log p(u^i_{j,k} | U^{i+1}_{k,·})
            %     = -0.5* ((u)^T C^{-1} u + log |2π C|)
            %       where C is pamameterized by its descendants or
            %       hyperparameters.
            %
            % Argument:
            %   u(x):   The vector input of pdf. Leave it to empty if you want 
            %           to use the estimate of MAP or HMC. x \in R^N
            % Return:
            %   val:    log p(u | U^{i+1}_{k,·})
            %
            if nargin < 2
                x = obj.dgp.x;
            end
            
            if nargin < 3
                u = obj.u;
            end
            
            % Check if there are descendants or just hyperparameters
            if isobject(obj.descendants.l)
                % Fetch son's value
                l = obj.descendants.l.u; 
            else
                l = obj.descendants.l;
            end
            
            if isobject(obj.descendants.sigma)
                sigma = obj.descendants.sigma.u; 
            else
                sigma = obj.descendants.sigma;
            end
            
            if obj.stationary
                C = obj.gp_ker(x, x, l, sigma, obj.g);
            else
                % Calculate C parameterzied by its son
                C = obj.gp_ker(x, x', l, l', sigma, sigma', obj.g);
            end

            val = -0.5 * (tools.inv_chol(u', C) * u + tools.log_det(2*pi*C));
        end
        
        function val = grad_log_pdf_u(obj, x, u)
            % Give the (num) gradient of log probability density w.r.t 
            % u, which is a quadratic form.
            %
            % val = ∂-log p(u^i_{j,k} | U^{i+1}_{k,·}) / ∂u^i_{j,k}
            %     = -C^{-1} u
            %
            % Argument:
            %   u(x):   The vector input of pdf. Leave it to empty if you want 
            %           to use the estimate of MAP or HMC.
            % Return:
            %   val:    A vector. same length with u.
            %
            
            if nargin < 2
                x = obj.dgp.x;
            end
            
            if nargin < 3
                u = obj.u;
            end 
            
            if isobject(obj.descendants.l)
                % Fetch son's value
                l = obj.descendants.l.u; 
            else
                l = obj.descendants.l;
            end
            
            if isobject(obj.descendants.sigma)
                sigma = obj.descendants.sigma.u; 
            else
                sigma = obj.descendants.sigma;
            end
            
            if obj.stationary
                C = obj.gp_ker(x, x, l, sigma, obj.g);
            else
                % Calculate C parameterzied by its son
                C = obj.gp_ker(x, x', l, l', sigma, sigma', obj.g);
            end
               
            val = -tools.invL_chol(C, u);
        end
        
        function grad = grad_log_pdf_ls(obj, x, u)
            % Give the (num) gradient of log probability density w.r.t 
            % the length scale l and sigma
            % Only for non-stationary covariance, as we don't train the
            % last layer hyperparameters here.
            %
            % val = ∂log p(u^i_{j,k} | U^{i+1}_{k,·}) / ∂l
            %     = ...
            %
            % Argument:
            %   u(x):   The vector input of pdf. Leave it to empty if you want 
            %           to use the estimate of MAP or HMC.
            % Return:
            %   val:    An array. [dlog/dℓ(1:N), dlog/dσ(1:N)]
            %
            
            if nargin < 2
                x = obj.dgp.x;
            end
            
            if nargin < 3
                u = obj.u;
            end 
            
            grad = zeros(length(x), 2);
            
            if isobject(obj.descendants.l)
                % Fetch son's value
                l = obj.descendants.l.u; 
            else
                l = ones(length(x), 1) .* obj.hyper_para(:, 1);
            end
            
            if isobject(obj.descendants.sigma)
                % Fetch son's value
                sigma = obj.descendants.sigma.u; 
            else
                sigma = ones(length(x), 1) .* obj.hyper_para(:, 2);
            end
                        
            C = obj.gp_ker(x, x', l, l', sigma, sigma', obj.g);
                        
            alp = tools.invL_chol(C, u);
            
            for i = 1:length(x)
                % Get dK/dl and dK/ds, which are Christian cross-like matrices
                [dCdl, dCds] = obj.grad_func(x, l, sigma, i, obj.g, obj.dg);

                grad(i, 1) = 0.5 * trace(alp * alp' * dCdl - tools.invL_chol(C, dCdl));
                grad(i, 2) = 0.5 * trace(alp * alp' * dCds - tools.invL_chol(C, dCds));
            end
            
            if ~isobject(obj.descendants.l)
                grad(:, 1) = NaN;
            end
            
            if ~isobject(obj.descendants.sigma)
                grad(:, 2) = NaN;
            end
            
        end
        
        %% Miscelenous Functions for the class property managements
        function set_hyper_para(obj, hyper_para)
            % Change the hyper parameters. Because Matlab can not pass 
            % reference of a variable, thus one needs to manually change
            obj.hyper_para = hyper_para;
            
            % As well as for descendants
            if ~isobject(obj.descendants.l)
                obj.descendants.l = hyper_para(:, 1);
            end
            
            if ~isobject(obj.descendants.sigma)
                obj.descendants.sigma = hyper_para(:, 2);
            end
        end
        
    end
end