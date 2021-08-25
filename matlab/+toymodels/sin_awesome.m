function [y] = sin_awesome(x, a, b)
%Sinusoidal signal with time-varying frequency.
% Input interval must be of exactly [0, 1].
%
% Zheng Zhao (c) 2019
% zz@zabemon.com
%

if nargin < 2
    a = 7*pi;
end

if nargin < 3
    b = 2*pi;
end

if nargin < 4
    c = 5*pi;
end

if nargin < 5
    d = 2;
end

y = sin(a*cos(b*x.^2).*x).^2 ./ abs(cos(c*x)+d);

y = y(:)';

end

