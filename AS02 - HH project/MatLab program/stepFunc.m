function y= stepFunc(domain,start,final,hight)
sympref('HeavisideAtOrigin',1);
y=hight*(heaviside(domain-start)-heaviside(domain-final));
sympref('HeavisideAtOrigin','default');
end