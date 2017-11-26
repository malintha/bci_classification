%ICA algorithm
function [W,Y] = ica(N,Y,W,Z, alpha, maxItr)
for i=1:maxItr
I = eye(size(Z,1),size(Z,1));
deltaW = (N*I - g(Y)*f(Y)') * W;
W = W + alpha*deltaW;
Y = W*Z;
fprintf('Iteration ID : %d \n',i);
disp(Y(1,1:5));
sizeA=size(Y,2);
disp(Y(1,(sizeA-5):sizeA));
end

function fx = f(x)
fx = x.^3;
end
function gx = g(x)
gx = tanh(x);
end
end

