%PCA dimention reduction
function[dimRedData] = pcaDim(newDim,data,eignVectors)
    OrderedEV = flip(flipud(eignVectors')');
    NewOrderedEV = OrderedEV(:,1:newDim);
    dimRedData=NewOrderedEV'*data;
end