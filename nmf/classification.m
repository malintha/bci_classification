clear;
data = load_data('k3b',0.6, 1,0);
xtr = data.Xtr;
ytr = data.Ytr;
xte = data.Xte;
yte = data.Yte;
[length_yte, ~] = size(yte);

% nmf(xtr, xte, basis_vecs, iterations_1, iterations_2, lower, upper)
% best results with nnmf (multiplicaltive update) and minje's update rule at 44/29 basis vecs
% best results with nnmf (als update) and minje's update rule at 27 basis vecs
% for i=35:50

[h_tr, h_te] = nmf(xtr, xte, 44, 1000, 1000, 2, 50);
% [acc_nmf1, acc_nmf2] = sl_nn(h_tr, ytr, h_te, yte, 0.002, 12000, 35);
[idx, c] = kmeans(h_te,4);
kmeans_acc = c./sum(c,1);
acc = kmeans_acc./max(kmeans_acc);
acc = acc == 1;

accuracyCalc()

% fprintf('Basis: %d Accuracy of the classification using train data: %f \n',i,acc_nmf1);
% fprintf('Basis: %d Accuracy of the classification using test data: %f \n',i,acc_nmf2);

% end
% [h_tr, h_te] =  plsi(xtr, xte, 40, 1000, 1000, 2, 30);
% [acc_nmf1, acc_nmf2] = sl_nn(h_tr, ytr, h_te, yte, 0.001, 30000, 35);
% fprintf('Basis: %d Accuracy of the classification using train data: %f \n',i,acc_nmf1);
% fprintf('Basis: %d Accuracy of the classification using test data: %f \n',i,acc_nmf2);

%%%%matlab classify%%%%%%%%
% class = classify(h_te', h_tr', ytr);
% y_hat_class = class == yte;
% corrects_classify = sum(y_hat_class);
% accuracy_classify = corrects_classify/length_yte*100;
% fprintf('Classify basis: %d correct: %d accuracy: %4f \n',i, corrects_classify, accuracy_classify);

%%%%%%%%%single layer nn%%%%%%%%%
% sl_nn(h_tr,ytr,h_te, yte,learningRate,iterations)
% [acc1, acc2] = sl_nn(h_tr, ytr, h_te, yte, 0.002, 20000, 35);
% fprintf('Basis: %d Accuracy of the classification using train data: %f \n',i,acc1);
% fprintf('Basis: %d Accuracy of the classification using test data: %f \n',i,acc2);

% end


function acc = accuracyCalc(Y,yPred)
[m,n]=size(Y);
count=0;
for i=1:n
    count=count+sum((Y(:,i)&yPred(:,i)));
end
acc=count/n*100;
end


% get the target matrix 4 rows given one grouping vector
function[y_out] = get_target_matrix(y)
[n_trials, ~] = size(y);
y_1 = ones(n_trials,1); y_2= ones(n_trials,1); y_3=ones(n_trials,1); y_4 = ones(n_trials,1);
y_1(:,1) = 1;
y_2(:,1) = 2;
y_3(:,1) = 3;
y_4(:,1) = 4;
y_1 = y_1==y;
y_2 = y_2==y;
y_3 = y_3==y;
y_4 = y_4==y;
y_out = horzcat(y_1, y_2, y_3, y_4);
end
