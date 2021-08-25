% Class for Deep Gaussian processes
%
% (poorly written)
%
% Zheng Zhao (c) 2019 
% zz@zabemon.com
%

classdef DGP < handle
    %This is an abstract class for DGP, a collection for DGPNode
    
    properties
        L               % Number of layers
        Li              % Numer of nodes in i-th layer (colum vector)
        
        nodes           % Collection of DGPNode in a cell {L, Li}
        U               % A matrix containing values of all nodes. This 
                        % should be done more efficiently using DGPNode.u
                        % properties of each node, but Matlab optimizer 
                        % does not support this feature.
                        %
                        % U = [f|1 u11|1 u12|1 u21|1 u22|1 ... ...
                        %      f|2 u11|2 u12|2 u21|2 u22|2 ... ...
                        %      ...                                
                        %      f|N u11|N u12|N u21|N u22|N ... ...]
                        %
        
        % Flag for telling if the DGP instance is compiled
        compiled
        
        % ss would be an SSDGP object, if linked
        ss
        
        % Data pairs
        x
        y
        R
        N
        data_loaded
        
        % Optimization history
        opt_history
    end
    
    methods
        %% Constructor and tools
        function obj = DGP(f)
            % Constructor and initialize
            %
            % Argument:
            %   f:      The all father node.
            %
            
            % Find all nodes and put them casedely in the cell "nodes"
            obj.L = 1;
            obj.Li = [1];
            obj.nodes{1, 1} = f;
            
            flag = obj.has_son(f);
            son = obj.find_son(f);
            
            while flag
                % A loop for layer
                obj.L = obj.L + 1;
                obj.Li = [obj.Li length(son)];
                for i = 1:obj.Li(end)
                   obj.nodes{obj.L, i} = son{i}; 
                end
                
                % update son
                son = {};
                for i = 1:obj.Li(end)
                    son = [son obj.find_son(obj.nodes{obj.L, i})];
                end
                
                if isempty(son)
                    flag = 0;
                end
            end

        end
        
        function load_data(obj, x, y, R)
            % Load data to obj.x and obj.y for regression
            % y = f(x)
            %
            % Arguments:
            %   x, y:   Data pair. Both in scalar column vector
            %   R:      The measurement noise covariance
            
            % Check if DGP is compiled
            if ~obj.compiled
                error('DGP is not compiled.')
            end
            
            % Ensure the data are column vectors
            x = x(:);
            y = y(:);
            
            % Ensure the dimension matches
            if numel(unique([length(x), length(x), size(R,1), size(R,2)])) ...
                    ~= 1
                error('Data dimension doe not match.');
            end
            
            obj.x = x;
            obj.y = y;
            obj.R = R;
            obj.N = length(x);
            
            % initialize U
            obj.U = zeros(obj.N, sum(obj.Li));
            
            obj.data_loaded = 1;
            
            fprintf('Data loaded. \n')
        end
        
        function compile(obj)
            % A function that check if everything is okay before loading
            % actual data and doing inference. This method is mandatory to
            % call.
            
            % Let the nodes 1. know the DGP container 2. know their father
            % 3. know its unique idx in U.  4. Check if names are unique
            if obj.compiled
                error('DGP is already compiled.');
            end
            
            name_collect = {};
            idx = 0;
            for i = 1:obj.L
                for j = 1:obj.Li(i)
                    idx = idx + 1;
                    obj.nodes{i, j}.dgp = obj;
                    obj.nodes{i, j}.compiled = 1;
                    obj.nodes{i, j}.U_idx = idx;
                    
                    if isobject(obj.nodes{i, j}.descendants.l)
                        obj.nodes{i, j}.descendants.l.father = obj.nodes{i, j};
                    end
                    
                    if isobject(obj.nodes{i, j}.descendants.sigma)
                        obj.nodes{i, j}.descendants.sigma.father = obj.nodes{i, j};
                    end
                    
                    name_collect = [name_collect obj.nodes{i, j}.name];
                end
            end
            
            if length(unique(name_collect)) ~= length(name_collect)
                error('Names of DGP nodes are not unique');
            end
            
            obj.compiled = 1;
            fprintf('DGP compiled. \n')
        end
        
        function flag = has_son(obj, A)
            % Tell if the node a has descendant(s)
            % A: a DGPNode object
            % obj has to be passed as an argument, otherwise will get
            % reference error. The same below. 
            %
            flag = isobject(A.descendants.l);
        end
        
        function flag = has_no_son(obj, A)
            % Tell if the node A has no sons
            % flag 1: has no son
            flag = isnumeric(A.descendants.l) & isnumeric(A.descendants.l);
        end
        
        function sons = find_son(obj, A)
            % Find all sons of a node and return as a cell
            % Here we only implement a special binary case of matern
            % covariance function.
            sons = {};
            if isobject(A.descendants.l)
                sons{1} = A.descendants.l;
            end
            if isobject(A.descendants.sigma)
                sons{2} = A.descendants.sigma;
            end
        end
        
        function assign_u(obj, U)
            % Assign the U(x_1:N) to each of the node.
            % Though this is stupid, but that's all I can do with Matlab.
            %
            idx = 0;
            for i = 1:obj.L
                for j = 1:obj.Li(i)
                    idx = idx + 1;
                    obj.nodes{i, j}.u = U(:, idx);
                end
            end
        end
        
        %% MAP Inferencer
        function [neg_log_likeli] = MAP(obj, f_bound, options)
            % Maximum a posteriori inference for DGP.
            % Use the built-in Matlab optimization toolbox.
            %
            % Argument:
            %   f_bound:    The lower and upper bound of f. Default is Inf
            %   options:    options object from Optimization toolbox
            %

            % Set optimization options
            if nargin < 3
                options = optimoptions('fmincon','Algorithm','Interior-Point', ...
                                    'HessianApproximation', 'lbfgs', ... 
                                    'SpecifyObjectiveGradient',true, 'Display', ...
                                    'iter-detailed', 'MaxIterations', 3000, ...
                                    'CheckGradients', false);
            end
            
            % Give lower bound and upper bound of constraint optimization,
            % the same size with obj.U elementwisely
            if nargin < 2
                f_min = -Inf;
                f_max = Inf;
            else
                f_min = f_bound(1);
                f_max = f_bound(2);
            end
            [lower, upper] = obj.opt_boundary(f_min, f_max);
            
            % Optimize. This function can not use the reference of
            % variables, thus has to copy estimate.
            [obj.U, neg_log_likeli] = fmincon(@obj.dgp_opt_handle, obj.U, ...
                            [], [], [], [], lower, upper, [], options);
            
            % Dispatch them to each of the nodes (not very necessry though,
            % )
            obj.assign_u(obj.U);
        end
        
        function [obj_val, grad] = dgp_opt_handle(obj, U)
            % A Matlab handler for giving the objective function value and
            % gradients of the DGP at current estimate U. 
            %
            % Argument:
            %   U:          The current estimate (value) of the DGP, at
            %               location x1,...xN
            % Return:
            %   obj_val:    Negative log of the posterior propotion. 
            %               L = -log [p(y(1:N) | U(1:N)) p(U(1:N))]
            %   grad:       -∂L/∂U (A matrix)
            %
            
            % Dispatch U to each of the nodes
            obj.assign_u(U);
            
            % Init grad
            grad = zeros(size(U, 1), size(U, 2));
            
            % loss for p(y|U) and grad for f
            obj_val = 0.5 * (tools.inv_chol((obj.y - U(:, 1))', obj.R) * (obj.y - U(:, 1)) ...
                         + tools.log_det(2*pi*obj.R));

            for i = 1:obj.L
                for j = 1:obj.Li(i)
                    obj_val = obj_val - obj.nodes{i, j}.log_pdf();
                end
            end
                     
            % grad for f
            grad(:, 1) = - tools.invL_chol(obj.R, (obj.y - U(:, 1))) ...
                         - obj.nodes{1, 1}.grad_log_pdf_u();
                     
            % quadratic grad for other GP nodes
            idx = 1;
            for i = 2:obj.L
                for j = 1:obj.Li(i)
                    idx = idx + 1;
                    grad(:, idx) = - obj.nodes{i, j}.grad_log_pdf_u();
                end
            end
            
            % non least square grad for other GP nodes
            idx = 0;
            for i = 1:obj.L
                for j = 1:obj.Li(i)
                    idx = idx + 1;
                    grad_ls = -obj.nodes{i, j}.grad_log_pdf_ls();
                    % Assign them accordingly
                    if ~isnan(grad_ls(:, 1))
                        idx_l = obj.nodes{i, j}.descendants.l.U_idx;
                        grad(:, idx_l) = grad(:, idx_l) + grad_ls(:, 1);
                    end
                    if ~isnan(grad_ls(:, 2))
                        idx_s = obj.nodes{i, j}.descendants.sigma.U_idx;
                        grad(:, idx_s) = grad(:, idx_s) + grad_ls(:, 2);
                    end
                end
            end
        end
        
        function [lower, upper] = opt_boundary(obj, f_min, f_max)
            % Give the optimization boundary for the estimates
            % This has to be done, otherwise the optimizer will diverge
            % The boundary for f should be individually set
            % Differnet g functions should have different bound.
            
            % Init
            lower = obj.U;
            upper = lower;
            
            % Set f
            lower(:, 1) = f_min * ones(obj.N, 1);
            upper(:, 1) = f_max * ones(obj.N, 1);
            
            % Set for the rest of the nodes. Note that the g comes from
            % their father
            idx = 1;
            for i = 2:obj.L
                for j = 1:obj.Li(i)
                    idx = idx + 1;
                    g_info = functions(obj.nodes{i, j}.father.g);
                    if strcmp(g_info.function, '@(l)exp(l)')
                        % Small length-scale suggested by Ezmir
                        lower(:, idx) = -15 * ones(obj.N, 1);
                        upper(:, idx) = 4 * ones(obj.N, 1);
                    elseif strcmp(g_info.function, '@(l)l.^2')
                        lower(:, idx) = -5 * ones(obj.N, 1);
                        upper(:, idx) = 5 * ones(obj.N, 1);
                    elseif strcmp(g_info.function, '@tools.exp_lin')
                        lower(:, idx) = -6 * ones(obj.N, 1);
                        upper(:, idx) = 20 * ones(obj.N, 1);
                    else
                        error('Unsupported function g().')
                    end
                end
            end
            
        end
        
        function stop = record_opt_history(obj, x, optimValues, state)
            % A standard Matlab output function for fmincon
            % https://se.mathworks.com/help/matlab/math/output-functions.html
            % Here mainly records the optimization history and stopping
            % criteria.
            %
            stop = false;
            
            switch state
                case 'init'
                      obj.opt_history = [];
                case 'iter'
                      obj.opt_history = [obj.opt_history optimValues.fval];
            end
        end
        
        function [i, j] = find_node_by_name(obj, name)
            % A function that finds the node index by using its name
            
            % Check if is compiled
            if ~obj.compiled
                error('DGP is not compiled.');
            end
            
            for i = 1:obj.L
                for j = 1:obj.Li(i)
                    if strcmp(obj.nodes{i, j}.name, name)
                        return
                    end
                end
            end
            
            i = NaN;
            j = NaN;
            
        end
        
        function val = read(obj, i, j, g)
            % A trivial reader function, just aims to simplify the upstream 
            % coding
            %
            % Arguments:
            %   i, j:   The j-th node of i-th layer
            %   g:      Boolean. Give the value by its g()?
            %
            if g && ~isempty(obj.nodes{i, j}.father)
                g = obj.nodes{i, j}.father.g;
            else
                g = @(x) x;
            end
            
            val = g(obj.U(:, obj.nodes{i, j}.U_idx));
        end
    end
end

