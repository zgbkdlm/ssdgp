function [X, Y, ind_query] = interp_x(x, y, query)
%Give integration/interpolation/prediction positions
% Different from directly using interp method of Matlab, here x might be
% non-temporal, which needs an additional sorting step.
%
% Arguments:
%   x:      x_{1:N} location where you got measurements.
%   y:      y(x_{1:N}) The measurements.
%   query:  x_{τ} for any τ you want to query.
%
% Return:
%   X:          x_{...} location including measuremnt and query point in
%               acending order.
%   Y:          Corresponding measurements. For query points, y=NaN.
%   ind_query:  Indices of X where there are query points
%
% Zheng Zhao 趙正 (c) 2019 
% zz@zabemon.com
%

% % Ensure x and y are colum
% x = x(:);
% query = query(:);
% y = y(:);

% Prepocessing of data ensuring there is duplicated points
if ~isempty(intersect(x, query))
    warning(sprintf('x and query have duplicated values: %s. \n Fixed by removing those duplicated points', ...
            num2str(intersect(x, query))));
    % We then remove the duplicated query point
    query(ismember(query, intersect(x, query))) = [];
end

data_ensemble = [x' query'; 
                 y' NaN*ones(1,length(query))];  % can be faster here

% Sorting data according to X
sort_data = sortrows(data_ensemble', 1, 'ascend')';
X = sort_data(1, :)'; 
Y = sort_data(2:end, :)'; 
ind_query = find(isnan(Y));

end

