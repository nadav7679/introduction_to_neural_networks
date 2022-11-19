function [x,y] = hVars(V)

alpha = 0.07*exp(-V./20);
beta = 1./(exp((30-V)./10)+1);
x = alpha./(alpha + beta);
y = 1./(alpha + beta);

end