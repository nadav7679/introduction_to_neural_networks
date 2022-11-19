domain = -40:0.3:120;
[hInf,hTau] = hVars(domain);
[nInf,nTau] = nVars(domain);
[mInf,mTau] = mVars(domain);
figure
plot(domain,hTau,'r',domain,nTau,'b',domain,mTau,'g-')
legend('h','n','m')
xlabel('Voltage (mV)')
ylabel('Time comstant (msec)')
figure
plot(domain,hInf,'r',domain,nInf,'b',domain,mInf,'g-')
legend('h','n','m')
xlabel('Voltage (mV)')
ylabel('Activation')