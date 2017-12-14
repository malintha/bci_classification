clear;
clc;
addpath('bin');
data= load_data('k3b', 0.4, 1,0);%k3b.mat k6b.mat l1b.mat

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
oneY=ones(size(oneSTFT,2),1);
twoY=2.*ones(size(twoSTFT,2),1);
threeY=3.*ones(size(threeSTFT,2),1);
fourY=4.*ones(size(fourSTFT,2),1);
%removing NaN
[dataT1,oneY]=removeNaN(oneSTFT,oneY);
[dataT2,twoY]=removeNaN(twoSTFT,twoY);
[dataT3,threeY]=removeNaN(threeSTFT,threeY);
[dataT4,fourY]=removeNaN(fourSTFT,fourY);
[dataTrain,Ytr]=removeNaN(dataTrain,Ytr);
[dataTest,Yte]=removeNaN(dataTest,Yte);

%NMF
basisVectors=40;
maxItr=1000;
learnMode=0;
wInit=[];
[E1, W1, H1] = nmf(dataT1,basisVectors,maxItr,learnMode,wInit);
[E2, W2, H2] = nmf(dataT2,basisVectors,maxItr,learnMode,wInit);
[E3, W3, H3] = nmf(dataT3,basisVectors,maxItr,learnMode,wInit);
[E4, W4, H4] = nmf(dataT4,basisVectors,maxItr,learnMode,wInit);

wInit=cat(2, W1, W2, W3, W4);
%training data
maxItr=1000;
learnMode=0;
[ETr, WTr, HTr] = nmf(dataTrain,4*basisVectors,maxItr,learnMode,wInit);
%figure, imagesc(HTr);
%colorbar();
maxItr=1000;
[ETe, WTe, HTe] = nmf(dataTest,4*basisVectors,maxItr,learnMode,wInit);
%figure, imagesc(HTe);
%colorbar();

%NN parameters
%%%%%Train NN using OXtr%%%%%%%%
X_inputTr = HTr;
X_inputTe = HTe;
Y_label =getOnehotVectorMode(Ytr);
learningRate=0.2   ;%learningRate=0.00001; inner layer 35
iterations=20000;%iterations=5000;

[W,b]=simpleNeuralNetwork(X_inputTr,Y_label,learningRate,iterations);

Ynew =seprateDataSimple(W,b,X_inputTr);
%accuracy calculation
acc = accuracyCalc(Y_label,Ynew);
fprintf('Accuracy of the classification using train data : %f \n',acc);

Ynew =seprateDataSimple(W,b,X_inputTe);
Ytest=getOnehotVectorMode(Yte);
%accuracy calculation
acc = accuracyCalc(Ytest,Ynew);
fprintf('Accuracy of the classification using test data : %f \n',acc);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%NN functions%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%NN functions%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%Simple neuralNetwork fuction
function [W,b] = simpleNeuralNetwork(Z,y,learningRate,iterations)
% W and B initialization
[m,n] = size(Z);
[p,q] = size(y);
b = ones(1, n);%bias
Z = vertcat(Z,b);
rng(0);
W = normrnd(0,1,p, m+1)*0.01;
for i=1:iterations
    Z2 = W*Z;
    YHot = softMax(Z2);
    dB = (YHot-y).*dsoftMax(Z2);%dB = (Z2-y);
    dW = Z*dB';
    W = W-(learningRate*dW');
    %Error calulated as cross entropy:
    error=sum(-sum(y.*log(YHot)));
    fprintf('Iteration : %d, squared error : %f \n',i, error);  
end    
end
%seperate Data Function
function Ynew = seprateDataSimple(W,b,Z)
%nor = ones(size(b,2),size(Z,2));
[m,n] = size(Z);
b = ones(1, n);
Z = vertcat(Z,b);
A = W*Z;%A = W*Z + b*nor;%A = W*Z + b(:,1:size(Z,2));
sfOut=softMax(A);
[p,q]=size(sfOut);
Ynew=zeros(p,q);
for i = 1:q
    [val,index]=max(sfOut(:,i));
    Ynew(index,i)=1;
end
end
%Accuracy Calulation
function acc = accuracyCalc(Y,yPred)
[m,n]=size(Y);
count=0;
for i=1:n
    count=count+sum((Y(:,i)&yPred(:,i)));
end
acc=count/n*100;
end
%differenctiation of softmax activation
function dsx = dsoftMax(x)
sxij = softMax(x);
dsx=sxij.*(1-sxij);
end
%softmax activation
function sx = softMax(x)   
sx= exp(x)./sum(exp(x));
end
%differenctiation of logistic function
function dgx = dg(x)
dgx = g(x).*(1-g(x));
end
%logistic function
function gx = g(x)
gx = 1.0./(1.0 + exp(-x));
end
%differenctiation of tanh function
function dgx = dg2(x)
dgx =1-g2(x).*g2(x);
end
%tanh function
function gx = g2(x)
gx = tanh(x);
end
