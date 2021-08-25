function [mid_line, poly, boundary] = errBar(x, y, err, varargin)
% Plot fancy error bar in one shot
%
% Arguments
%   x, y:       Data pair. 
%   err:        The error
%   varargin:   The properties for the line, fill, and boundaries
%               {...}, the paras for the middle line
%               {...}, the paras for the polygon
%               {...}, the paras for the top boundary line
%               {...}, the paras for the bot boundary line (optional)
%
% Return:
%   bar:
%
% Zheng Zhao @ 2019
% zz@zabemon.com
%

property_line = varargin{1};
property_patch = varargin{2};

boundary = {};

if nargin < 6
    no_boundaries = true;
else
    property_top = varargin{3};
    if nargin < 7
    property_bot = property_top;
    else
        property_bot = varargin{4};
    end
end

% Force colume vectors
x = x(:);
y = y(:);
err = err(:);

poly_x = [x; flip(x, 1)];
ploy_y = [y+err; flip(y-err, 1)];
face = [1:1:length(poly_x)];

poly = patch('Faces', face, 'Vertices', [poly_x ploy_y], property_patch{:});
hold on

% Plot the data line
mid_line = plot(x, y, property_line{:});

% Plot the error boundaries
if ~no_boundaries
    
    top = plot(x, y+err, property_top{:});
    bot = plot(x, y+err, property_bot{:});
    
    boundary = {top, bot};

end

end

