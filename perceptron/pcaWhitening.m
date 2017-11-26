%PCA whitening
function[dimRedData] = pcaWhitening(newDim,data)  
[V, D] = eignVectors(data);
OrderedEV = flip(flipud(V')');
NewOrderedEV = OrderedEV(:,1:newDim);
OrderedD = flip(flipud(D')');
%data=flip(flipud(data')');
DNew = OrderedD(1:newDim,:);
NewOrderedD = DNew(:,1:newDim);
dimRedData=sqrt(inv(NewOrderedD))*NewOrderedEV'*data;
end