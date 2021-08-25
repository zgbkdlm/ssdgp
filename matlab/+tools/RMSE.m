function [val] = RMSE(x1, x2, varargin)
% Give the Root Mean Square Error
% Input:
%     x1, x2: n*m matrix, where n and m is dim of state and observation.
%     norm:   "norm" will give normalized RMSE
% Output:
%     val:    RMSE with size n*1

if size(varargin, 1) ~= 0
    val = sqrt(1 / size(x1, 2) * sum((x1 - x2).^2, 2)) ./ (max(x2, [], 2) - min(x2, [], 2));
else
    val = sqrt((1 / size(x1, 2)) * sum((x1 - x2).^2, 2));
end

end

