clear;
clc;
addpath('bin');
data= load_data('k3b', 0.6, 1,0);%k3b.mat k6b.mat l1b.mat

%parameter initialization
Xte=data.Xte;
Xtr=data.Xtr;
Yte=data.Yte;
Ytr=data.Ytr;

frameSize = 128;
hopSize=64;
lowerFr=2;%lowerFr=2;
upperFr=30;%upperFr=257;

data= seperateClasses(Xtr,Ytr);
one=data.one;
two=data.two;
three=data.three;
four=data.four;

%Mu = 3-7, lowerFr=6Hz upperFr=14Hz,  Mu+Beta = 3-15,lowerFr=6Hz upperFr=30Hz ,ony set numbers which are devisible by 2
oneSTFT = abs(dostft(one, frameSize, hopSize,'blackman',lowerFr,upperFr));
twoSTFT = abs(dostft(two, frameSize, hopSize,'blackman',lowerFr,upperFr));
threeSTFT = abs(dostft(three, frameSize, hopSize,'blackman',lowerFr,upperFr));
fourSTFT = abs(dostft(four, frameSize, hopSize,'blackman',lowerFr,upperFr));
dataTrain = abs(dostft(Xtr, frameSize, hopSize,'blackman',lowerFr,upperFr));
dataTest =  abs(dostft(Xte, frameSize, hopSize,'blackman',lowerFr,upperFr));
oneY=ones(size(oneSTFT,2));
twoY=2.*ones(size(twoSTFT,2));
threeY=3.*ones(size(threeSTFT,2));
fourY=4.*ones(size(fourSTFT,2));
%removing NaN
[dataT1,oneY]=removeNaN(oneSTFT,oneY);
[dataT2,twoY]=removeNaN(twoSTFT,twoY);
[dataT3,threeY]=removeNaN(threeSTFT,threeY);
[dataT4,fourY]=removeNaN(fourSTFT,fourY);
[dataTrain,Ytr]=removeNaN(dataTrain,Ytr);
[dataTest,Yte]=removeNaN(dataTest,Yte);

%NMF
basisVectors=1;
maxItr=2000;
learnMode=0;
wInit=[];
[E1, W1, H1] = nmf(dataT1,basisVectors,maxItr,learnMode,wInit);
[E2, W2, H2] = nmf(dataT2,basisVectors,maxItr,learnMode,wInit);
[E3, W3, H3] = nmf(dataT3,basisVectors,maxItr,learnMode,wInit);
[E4, W4, H4] = nmf(dataT4,basisVectors,maxItr,learnMode,wInit);

wInit=cat(2, W1, W2, W3, W4);
maxItr=200;
noOfClasses=4;
learnMode=1;
[ETr, WTr, HTr] = nmf(dataTrain,noOfClasses*basisVectors,maxItr,learnMode,wInit);
maxItr=1000;
[ETe, WTe, HTe] = nmf(dataTest,noOfClasses*basisVectors,maxItr,learnMode,wInit);

y_hatTr = getActivationClass(HTr,basisVectors,noOfClasses);
y_hatTe = getActivationClass(HTe,basisVectors,noOfClasses);

%figure, imagesc(WTr);
%colorbar();
%figure, imagesc(HTr);
%colorbar();


%accuracy calculation
YTrain =getOnehotVectorMode(Ytr);
acc = accuracyCalc(YTrain,y_hatTr);
fprintf('NMF Accuracy of the classification using train data : %f \n',acc);

Ytest=getOnehotVectorMode(Yte);
%accuracy calculation
acc = accuracyCalc(Ytest,y_hatTe);
fprintf('NMF Accuracy of the classification using test data : %f \n',acc);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%NN functions%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%change labels to one hot vector mode
function y_hat = getActivationClass(H,basisVectors,classes)
% identify unique classes from labels
y=[];
index=1;
for i = 1:classes
    y1= sum(H(index:index+basisVectors-1,:),1);
    y=cat(1, y, y1);
   index=index+basisVectors;
end
y_hat = y./max(y);
y_hat = y_hat == 1;
end
%change labels to one hot vector mode
function y = getOnehotVectorMode(y)
% identify unique classes from labels
[m,n]=size(y);
uniqueLabels = unique(y);
uniqueLabelsSize = length(uniqueLabels);
oneHot=zeros(m,uniqueLabelsSize);
for i = 1:m
    oneHot(i,y(i,n)) = 1;
end
y=oneHot';
end
%get Onehot vector

%Accuracy Calulation
function acc = accuracyCalc(Y,yPred)
[m,n]=size(Y);
count=0;
for i=1:n
    count=count+sum((Y(:,i)&yPred(:,i)));
end
acc=count/n*100;
end

