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

%Mu = 3-7, lowerFr=6Hz upperFr=14Hz,  Mu+Beta = 3-15,lowerFr=6Hz upperFr=30Hz ,ony set numbers which are devisible by 2
dataTrain = abs(dostft(Xtr, frameSize, hopSize,'blackman',lowerFr,upperFr));
dataTest =  abs(dostft(Xte, frameSize, hopSize,'blackman',lowerFr,upperFr));

%NN parameters
%%%%%Train NN using OXtr%%%%%%%%
X_input = dataTrain; 
Y_label =getOnehotVectorMode(Ytr);
learningRate=0.000001;%learningRate=0.000001;
iterations=100000;

[W,b]=simpleNeuralNetwork(X_input,Y_label,learningRate,iterations);

X_input = dataTrain;
Ynew =seprateDataSimple(W,b,X_input);
%accuracy calculation
acc = accuracyCalc(Y_label,Ynew);
fprintf('Accuracy of the classification using train data : %f \n',acc);

X_input = dataTest;
Ynew =seprateDataSimple(W,b,X_input);
Ytest=getOnehotVectorMode(Yte);
%accuracy calculation
acc = accuracyCalc(Ytest,Ynew);
fprintf('Accuracy of the classification using test data : %f \n',acc);

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
rng('default');
W = normrnd(0,1,p, m+1)*0.001;
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
