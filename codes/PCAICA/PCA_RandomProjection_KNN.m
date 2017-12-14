clear;
clc;
addpath('bin');
data= load_data('k3b', 0.5, 1,0);
MLim=50;
LLim=50;
KLim=50;
calculateAccuracy(data,MLim,LLim,KLim);

%%%%%%%%%%%%%functions for K
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%NEW Functions starts from here.

function calculateAccuracy(data,MLim,LLim,KLim)

%parameter initialization
Xte=data.Xte;
Xtr=data.Xtr;
Yte=data.Yte;
Ytr=data.Ytr;
frameSize = 128;
hopSize=64;
lowerFr=6;%lowerFr=2;
upperFr=30;%upperFr=257;

%Mu = 3-7, lowerFr=6Hz upperFr=14Hz,  Mu+Beta = 3-15,lowerFr=6Hz upperFr=30Hz ,ony set numbers which are devisible by 2
dataTrain = abs(dostft(Xtr, frameSize, hopSize,'blackman',lowerFr,upperFr));
dataTest =  abs(dostft(Xte, frameSize, hopSize,'blackman',lowerFr,upperFr));

[VTrain, DTrain] = eignVectors(dataTrain);
[VTest, DTest] = eignVectors(dataTest);

%calculateAccuracy(y,M,L,K);
maxacc=0;
optK=0;
optM=0;
optL=0;
minacc=100;
badK=0;
badM=0;
badL=0;
%Experiment running for M=2 to M=25, L=2 to L=25 and K=1 to K=50.
fid = fopen('experimentResults.txt','wt');
fprintf(fid,'M, L, K, accuracy\n');
for M = 2:MLim
    
    %PCA dimention reduction
    dimRedDataTrain = pcaDim(M,dataTrain,VTrain);
    dimRedDataTest = pcaDim(M,dataTest,VTest);
    
    for L = 2:LLim
        newDim=L;%random projection dim
        %further dimension change with random projection vector matrix A
        A = getRandomProjectionMatrix(newDim,M);
        %ATest = getRandomProjectionMatrix(newDim,selectedDimTest);
        randomProjectedDataTrain=sign(A*dimRedDataTrain);
        randomProjectedDataTest=sign(A*dimRedDataTest);
        
        for K = 2:KLim 
            %KNN algo
            %yTestCalculated = KNNclassification(dimRedDataTrain, dimRedDataTest, yTrain,K);
            yTestCalculated = KNNclassification(randomProjectedDataTrain, randomProjectedDataTest, Ytr,K);

            %accuracy calculation
            alogical = yTestCalculated == Yte;
            accuracy = double(sum(alogical))/size(Yte,1) *100;
            %fprintf('K(neighbors): %d,  L(Random project): %d and M_train: %d, M_test: %d, Accuracy: %f \n',K,newDim, selectedDimTrain,selectedDimTest,accuracy);
            fprintf('K(neighbors): %d,  L(Random project): %d and M(PCA dim red) : %d, Accuracy: %f \n',K,L,M,accuracy);
            fprintf(fid,'%d, %d, %d, %f\n',M,L,K,accuracy);
            if maxacc< accuracy
                   maxacc=accuracy;
                   optK=K;
                   optM=M;
                   optL=L;
             end
             if minacc > accuracy
                   minacc=accuracy;
                   badK=K;
                   badM=M;
                   badL=L;
             end
        end
    end   
end
fprintf(fid,'Max accuracy\n');
fprintf(fid,'%d, %d, %d, %f\n',optM,optL,optK,maxacc);
fprintf(fid,'Min accuracy\n');
fprintf(fid,'%d, %d, %d, %f\n',badM,badL,badK,minacc);
fclose(fid);

end

function [class_out] = KNNclassification(xTrain, xTest, yTrain, K)
% Calculate Hamming distance between eachtrain data and each test data
hammingDist = getHammingDistance(xTrain, xTest);
[sorted, indexes] = sort(hammingDist);
%choose 1st K points.
indexes = indexes(1:K,:);
knn_class_list = yTrain(indexes);

% identify unique classes from labels
uniqueLabels = unique(yTrain);
uniqueLabelsSize = length(uniqueLabels);
for i = 1:uniqueLabelsSize
    sumVals(i,:) = sum(knn_class_list == uniqueLabels(i));
end
% get max class label index with max values
[maximumVals, selectedClasses] = max(sumVals, [], 1);
% get class label
class_out = uniqueLabels(selectedClasses);
end

function distanceMat = getHammingDistance(DTrain, DTest)
%Distance is Hamming distance. 
distanceMat= pdist2(DTrain',DTest','hamming');
%Distance is 'euclidean'. 
%distanceMat= pdist2(abs(DTrain'),abs(DTest'));
end

%get random projection mat
function [A] = getRandomProjectionMatrix(newDim,PcaDim)
rng('default');
v = randn(newDim,PcaDim);
vsqrt=sqrt(sum(v'.^2));
A=v./vsqrt';
%te=norm(A(1,:));
end

