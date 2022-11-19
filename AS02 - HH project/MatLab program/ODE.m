function dy = ODE (t,y,T,I)
format long

I = interp1(T,I,t);
dy=zeros(4,1);

Gk=36;Gna=120;Gl=0.3; C=1;
Ek=-12 ; Ena=115 ; El=0; %Constants defined relative to resting potential

% y(1) = V , y(2) = n , y(3) = h , y(4) = m

[hInf,hTau] = hVars(dy(1));
[nInf,nTau] = nVars(dy(1));
[mInf,mTau] = mVars(dy(1));

dy(1)=(Gk/C)*(y(2)^4)*(y(1)-Ek) + Gna*(y(4)^3)*y(3)*(y(1)-Ena)+Gl*(y(1)-El)+I;
dy(2)= (nInf-y(2))/nTau;
dy(3)= (hInf-y(3))/hTau;
dy(4)= (mInf-y(4))/mTau;
