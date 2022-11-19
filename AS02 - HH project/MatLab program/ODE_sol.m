clc; close all; clear all;
timeSpan=[0 200];
IC = [0;0;0;0];
domain = 0:0.5:200;
I = stepFunc(domain,10,20,1);
[t,y]= ode45(@(t,y) ODE(t,y,domain,I),timeSpan,IC);
plot(t,y(:,1))
