function y = rect(x)
%Magnitude-varying rectangle wave.
% Input interval must be of exactly [0, 1].
%
% Zheng Zhao (c) 2019
% zz@zabemon.com
%

t1 = 1/6;
t2 = 2/6;
t3 = 3/6;
t4 = 4/6;
t5 = 5/6;

y((x>=0)&(x<t1)) = 0;
y((x>=t1)&(x<t2)) = 1;
y((x>=t2)&(x<t3)) = 0;
y((x>=t3)&(x<t4)) = 0.6;
y((x>=t4)&(x<t5)) = 0;
y((x>=t5)) = 0.4;

end

