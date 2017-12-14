clear;
data = load_data('l1b',0.6, 1,0);
xtr = data.Xtr;
ytr = data.Ytr;
xte = data.Xte;
yte = data.Yte;
[length_yte, ~] = size(yte);

% nmf(xtr, xte, basis_vecs, iterations_1, iterations_2, lower, upper)
% best results with nnmf (multiplicaltive update) and minje's update rule at 44/29 basis vecs
% best results with nnmf (als update) and minje's update rule at 27 basis vecs
i = 44;
[h_tr, h_te] = nmf(xtr, xte, 44, 1000, 1000, 2, 50);
[acc_nmf1, acc_nmf2] = sl_nn(h_tr, ytr, h_te, yte, 0.002, 12000, 35);
fprintf('NMF classification results using a single layer NN with 35 hidden units\n');
fprintf('Basis: %d Accuracy of the classification using train data: %f \n',i,acc_nmf1);
fprintf('Basis: %d Accuracy of the classification using test data: %f \n\n',i,acc_nmf2);

[h_tr, h_te] =  plsi(xtr, xte, i, 1000, 1000, 2, 30);
[acc_1, acc_2] = sl_nn(h_tr, ytr, h_te, yte, 0.001, 30000, 35);
fprintf('PLSI classification results using a single layer NN with 35 hidden units\n');
fprintf('Basis: %d Accuracy of the classification using train data: %f \n',i,acc_1);
fprintf('%f \n',acc_2);