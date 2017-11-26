clear;
clc;
data= load_data('k3b', 0.5, 1);%k3b.mat k6b.mat l1b.mat

%parameter initialization
Xte=data.Xte;
Xtr=data.Xtr;
Yte=data.Yte;
Ytr=data.Ytr;
Ytrain=getOnehotVectorMode(Ytr);
Ytest=getOnehotVectorMode(Yte);
frameSize = 128;
hopSize=64;
lowerFr=2;%lowerFr=2;
upperFr=30;%upperFr=257;

%Mu = 3-7, lowerFr=6Hz upperFr=14Hz,  Mu+Beta = 3-15,lowerFr=6Hz upperFr=30Hz ,ony set numbers which are devisible by 2
dataTrain = abs(dostft(Xtr, frameSize, hopSize,'blackman',lowerFr,upperFr));
dataTest =  abs(dostft(Xte, frameSize, hopSize,'blackman',lowerFr,upperFr));

%Matlab NN function
net = feedforwardnet(10);
net = train(net,dataTrain,Ytrain);
view(net)
y = net(dataTest);
perf = perform(net,y ,Ytest);
% get accuracy
% normalize y
y_hat = y./max(y);
y_hat = y_hat == 1;
%accuracy calculation
acc = accuracyCalc(y_hat,Ytest);
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
%Accuracy Calulation
function acc = accuracyCalc(Y,yPred)
[m,n]=size(Y);
count=0;
for i=1:n
    count=count+sum((Y(:,i)&yPred(:,i)));
end
acc=count/n*100;
end

