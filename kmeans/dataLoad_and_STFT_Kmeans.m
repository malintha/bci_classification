clear;
clc;
rawData = load('k3b.mat');
s=rawData.s;
HDR=rawData.HDR;
TRIG=HDR.TRIG;
Classlabel=HDR.Classlabel;
SampleRate=HDR.SampleRate;
data=preProcessData(s, TRIG, Classlabel,SampleRate);
XTrain=data.XTrain;
YTrain=data.YTrain;
%XTest=data.XTest; This doesnt have labels, so accuracy comparison is not possible
percentage=0.5;
data = seperateData(XTrain, YTrain, percentage);
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
frameSize = 64;
hopSize=48;

dataTrain = reformatData(Xtr, frameSize, hopSize);
dataTest = reformatData(Xte, frameSize, hopSize);

[VTrain, DTrain] = getEigenVect(dataTrain);
[VTest, DTest] = getEigenVect(dataTest);

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
    selectedDimTrain=M;%PCA dim train
    DOrderedTrain = flip(flipud(DTrain')');
    VOrderedTrain = flip(flipud(VTrain')');
    XOrderedTrain = flip(flipud(dataTrain')');
    VNewTrain = VOrderedTrain(:,1:selectedDimTrain);
    DNewTrain = DOrderedTrain(1:selectedDimTrain,:);
    DNewTrain = DNewTrain(:,1:selectedDimTrain);
    dimRedDataTrain=VNewTrain'*dataTrain;

    selectedDimTest=M;%PCA dim test
    DOrderedTest = flip(flipud(DTest')');
    VOrderedTest = flip(flipud(VTest')');
    XOrderedTest = flip(flipud(dataTest')');
    VNewTest = VOrderedTest(:,1:selectedDimTest);
    DNewTest = DOrderedTest(1:selectedDimTest,:);
    DNewTest = DNewTest(:,1:selectedDimTest);
    dimRedDataTest=VNewTest'*dataTest;
    
    for L = 2:LLim
        newDim=L;%random projection dim
        %further dimension change with random projection vector matrix A
        A = getRandomProjectionMatrix(newDim,selectedDimTrain);
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

%get eign vectors
function [A] = getRandomProjectionMatrix(newDim,PcaDim)
rng('default');
v = randn(newDim,PcaDim);
vsqrt=sqrt(sum(v'.^2));
A=v./vsqrt';
%te=norm(A(1,:));
end

%get eign vectors
function [V, D] = getEigenVect(X)
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
%%%%%%%%%%%%%%functions for KNN classification over%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Functions starts from here.
%seperate data
function data = preProcessData(s, TRIG, Classlabel,SampleRate)

%taking only c3=28, cz=31 and c4=34 channels
relatedS=[s(:,28),s(:,31),s(:,34)];
[m,n] = size(relatedS);
[p,q] = size(TRIG);

%seperating train and test data
testDataCount=sum(isnan(Classlabel));
trainDataCount=p-testDataCount;

XTest=zeros(SampleRate*4,n,testDataCount);
XTrain=zeros(SampleRate*4,n,trainDataCount);
YTrain=zeros(n,1);
testCount=1;
trainCount=1;

for i=1:p 
   sampleWindowLower= TRIG(i,1)+SampleRate*3 +1;%triger is at t=3secs
   sampleWindowUpper= TRIG(i,1)+SampleRate*7;%trail ends at t=7secs
   singleSample = relatedS(sampleWindowLower:sampleWindowUpper,:);
   y=Classlabel(i,1);
   if isnan(y)
       XTest(:,:,testCount)=singleSample;
       testCount=testCount+1;
   else 
       XTrain(:,:,trainCount)=singleSample;
       YTrain(trainCount,1)=y;
       trainCount=trainCount+1;
   end
end
data.XTrain=XTrain;
data.YTrain=YTrain;
data.XTest=XTest;
end
%reformat data and appy STFT
function [data] = reformatData(X, frameSize, hopSize)
[SR ,C, N] = size(X);
data = [];
%calculationg FR and FI parts
Fr = DFTR(frameSize);
Fi = DFTI(frameSize);
F= Fr-1i*Fi;
for i=1:N 
    Xtemp = X(:,:,i);
    C3 = Xtemp(:,1);
    CZ = Xtemp(:,2);
    C4 = Xtemp(:,3);
    %Creating the all frames.
    C3X = createX(C3,frameSize,hopSize);
    CZX = createX(CZ,frameSize,hopSize);
    C4X = createX(C4,frameSize,hopSize);
    %calculatng STFT
    FXC3=F*C3X;
    FXCZ=F*CZX;
    FXC4=F*C4X;
    
    %Discarding complex conjugates.
    SC3 = FXC3(1:(size(FXC3,1)/2+1),:);
    SCZ = FXCZ(1:(size(FXCZ,1)/2+1),:);
    SC4 = FXC4(1:(size(FXC4,1)/2+1),:);
    
    %Mu+Beta = 3-15   , Mu = 3-7 (with 64 framesize)
    C3_MuWaves = abs(SC3(3:7,:));
    CZ_MuWaves = abs(SCZ(3:7,:));
    C4_MuWaves = abs(SC4(3:7,:));
    
    C3_Y = reshape(C3_MuWaves,[size(C3_MuWaves,1)*size(C3_MuWaves,2) 1]);
    CZ_Y = reshape(CZ_MuWaves,[size(CZ_MuWaves,1)*size(CZ_MuWaves,2) 1]);
    C4_Y = reshape(C4_MuWaves,[size(C4_MuWaves,1)*size(C4_MuWaves,2) 1]);
    
    all_vect = [C3_Y; CZ_Y; C4_Y];
    data = [data all_vect];
       
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%OLD Functions starts from here.
%This function creates real parts of DFT:
function R = DFTR(size)
R = [];
for n= 0:size-1
    for f=0:size-1       
        R(n+1,f+1) = cos(2.0*pi*n*f/size);     
    end 
end
end
%This function creates imaginary parts of DFT:
function I = DFTI(size)
I = [];
for n= 0:size-1
    for f=0:size-1
        I(n+1,f+1) = sin(2.0*pi*n*f/size);     
    end 
end
end
%This function creates X matrix:
function X = createX(y,frameSize,hopSize)
X=[];
datasize = size(y,1);
remainder = rem(datasize,frameSize);
addZero=zeros(frameSize-remainder,1);%Sampling rate : frameSize. So we will have to add additional frameSize-remainder elements sound vector.
y = vertcat(y,addZero);
datasize = size(y,1);
totalWindowCount = 1+ ((datasize-frameSize)/(frameSize-hopSize));
startP =1;
endP= frameSize;
for i=1:totalWindowCount   
    dataChunk = y(startP:endP,:);
    dataChunkBlack = addBlackmanWindow(dataChunk);
    X=[X dataChunkBlack];   
    startP = startP + frameSize-hopSize;
    endP = endP + frameSize-hopSize;
end   
end
%This function creates apply hannWindow:
function dataChunkBlackman = addBlackmanWindow(dataChunk)
chunkSize = size(dataChunk,1);
dataChunkBlackman = blackman(chunkSize).*dataChunk;
end
%This function seperate data into two portions:
function data = seperateData(X, Y, percentage)
[p,q] = size(Y);
portion=round(p*percentage);
%seperating train and test data
Xte=X(:,:,1:portion);
Xtr=X(:,:,(portion+1):p);
Yte=Y(1:portion,1);
Ytr=Y(portion+1:p,1);
data.Xte=Xte;
data.Xtr=Xtr;
data.Yte=Yte;
data.Ytr=Ytr;
end

