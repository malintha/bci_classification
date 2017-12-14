%NMF gradient_descent function.
function [E, W, H] = nmf(X,basisVectors,maxItr,learnMode,wInit)
X = abs(X);
rng(0);%rng('default');
[m,n]=size(X);
e = 0.001;
W =  rand(m,basisVectors)*e;
H =  rand(basisVectors,n)*e;
if learnMode
   W = wInit;
end
X_hat_Init=W*H;
X_hat=X_hat_Init;
e = 0.00000001;
for i=1:maxItr
E=sum(sum((X.*log(X./X_hat))-X+X_hat));
fprintf('Iteration %d , Error : %f \n',i,E);
% update rules
oneFT=ones(size(X,1),size(X,2));
if ~learnMode
    W = (W .* ((X./((W*H)+e))*H' )./ ((oneFT*H')+e));
end
H = (H .* (W'*(X./((W*H) +e))) ./ ((W'*oneFT)+e));
X_hat=W*H;
end
end