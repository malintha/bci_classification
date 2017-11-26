%get eign vectors
function [V, D] = eignVectors(X)
%calculating mean for each row vector
%if nan values found replace with 0
X(isnan(X))=0;
dataM = mean(X);
%Subtract the mean
X=X-dataM;
%get covariance matrix
Xcov = cov(X');
%Xcov=X*X';
%get eigen matrix
[V,D] = eig(Xcov);
end