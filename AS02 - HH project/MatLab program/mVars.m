function [x,y]=mVars(V)

alpha = (25-V)./(10.*(exp((25-V)./10)-1));
beta = 4*exp(-V/18);
x = alpha./(alpha + beta);
y = 1./(alpha + beta);
end