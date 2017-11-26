%remove NaN present in STFT output
function [X, Y] = removeNaN(X,Y)
[row, col] = find(isnan(X));
U=unique(col);
m=size(U,1);
for i=1:m
    X(:,U(i,1)) = [];
    Y(U(i,1),:)=[];
end
end