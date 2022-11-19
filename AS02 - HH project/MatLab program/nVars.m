function [x,y]=nVars(V)
% This vector function returns n_infinity and tau_n for a given V.
% n_inf is y(1;1) and tau_n is y(2;1)

alpha = (10-V)./(100*(exp((10-V)./10)-1));
beta = 0.125.*exp(-V./80);
x = alpha./(alpha + beta);
y = 1./(alpha + beta);
end