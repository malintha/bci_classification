function[] = onlyNNforNMFoutput(HTr,HTe,Ytr, Yte)

%NN parameters
%%%%%Train NN using OXtr%%%%%%%%
X_inputTr = HTr;
X_inputTe = HTe;
Y_label =getOnehotVectorMode(Ytr);
learningRate=0.002   ;%learningRate=0.00001; inner layer 35
iterations=11000;%iterations=5000;

%figure, imagesc(X_inputTr);
%colorbar();
%figure, imagesc(X_inputTe);
%colorbar();
parameter=multilayerNeuralNetwork(X_inputTr,Y_label,learningRate,iterations,X_inputTe,getOnehotVectorMode(Yte));

Ynew =seprateData(parameter,X_inputTr);
%accuracy calculation
acc = accuracyCalc(Y_label,Ynew);
fprintf('Accuracy of the classification using train data : %f \n',acc);

Ynew =seprateData(parameter,X_inputTe);
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
%multilayer perceptron
function parameter = multilayerNeuralNetwork(Z,y,learningRate,iterations,dataTest,YN)
% W and B initialization

[m,n] = size(Z);
[p,q] = size(y);
b = ones(1, n);
Z = vertcat(Z,b);
X1 = Z;
middlelayerSize=35;
rng(0);%rng('default');
%1-50 weights for unput features, 51st weight for bias
W1 = normrnd(0,1,middlelayerSize, m+1);
%50 weights for unput to 2nd layer, 51th weight for bias
W2 = normrnd(0,1,p, middlelayerSize+1); 
maxTestAcc=0;
recorderedIter=0;
for i=1:iterations
    %forward propagation in 1st layer
    Z1 = W1*X1;%Z1 = W1*X1 + b1;
    X2 = g2(Z1); 
    %Adding bias for 2nd layer
    X2 = vertcat(X2,b); 
    %forward propagation in 2nd layer
    Z2 = W2*X2;% Z2 = W2*X2 + b2;
    YHat = softMax(Z2);
    % backpropagation in 2nd layer
    dB2 = (YHat - y).*dsoftMax(Z2);%This also works dB2 = (Z2 - y);%
    dW2 = X2*dB2';
    % backpropagation in 1st layer
    nW2=W2(:,1:size(W2,2)-1);
    dB1 = nW2'*dB2.*dg2(Z1);
    dW1 = X1*dB1';
    %learning parameters
    W1 = W1-(learningRate*dW1');
    W2 = W2-(learningRate*dW2');
    %Error calulated as cross entropy:
    error=sum(-sum(y.*log(YHat)));
    fprintf('Iteration : %d, squared error : %f \n',i, error);  
    Ynew = getOneHotVec(YHat);
    acc = accuracyCalc(Ynew,y);
    parameter.W1 = W1;
    parameter.W2 = W2;
    Ynew = seprateData(parameter,dataTest);
    accNew = accuracyCalc(Ynew,YN);
    if accNew>maxTestAcc
        recorderedIter=i;
        maxTestAcc=accNew;
    end
    if acc==100.0
        break;
    end
end
fprintf('Iteration : %d, max acc : %f \n',recorderedIter, maxTestAcc);  
end
%seperate Data Function
function Ynew = seprateData(parameter,Z)

[~,n] = size(Z);
b = ones(1, n);
%1st layer bias
Z = vertcat(Z,b);
X1 = Z;
W1=parameter.W1;
W2=parameter.W2;
%forward propagation in 1st layer
Z1 = W1*X1;
X2=g2(Z1);
%2ndlayer bias
X2 = vertcat(X2,b);
%forward propagation in 2nd layer
Z2 = W2*X2;
yHat=softMax(Z2);
[p,q]=size(yHat);
Ynew=zeros(p,q);
for i = 1:q
    [val,index]=max(yHat(:,i));
    Ynew(index,i)=1;
end
end
%get Onehot vector
function Ynew = getOneHotVec(Y)
[p,q]=size(Y);
Ynew=zeros(p,q);
for i = 1:q
    [val,index]=max(Y(:,i));
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

end