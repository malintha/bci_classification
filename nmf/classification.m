% clear;
% data = load_data('k3b',0.6, 1);
% xtr = data.Xtr;
% ytr = data.Ytr;
% xte = data.Xte;
% yte = data.Yte;
% [length_yte, ~] = size(yte);
% 
% %[h_tr, h_te] = nmf(xtr, xte, basis_vecs, iterations_1, iterations_2, lower, upper)   
% [h_tr, h_te] =  nmf(xtr, xte, 44, 1000, 1000, 6, 14);
% rng(1);

%%%%%%%%%%%%classifications%%%%%%%%%%%%%%%

%%%%%%%%%single layer nn%%%%%%%%%
y_target = get_target_matrix(ytr);
singlenet = feedforwardnet(30);
singlenet = train(singlenet,h_tr, y_target');
y = singlenet(h_te);
y_hat = y./max(y);
y_hat = y_hat == 1;
y_test = get_target_matrix(yte);
y_accuracy_test = y_hat'.*y_test;
accuracy = sum(sum(y_accuracy_test));
fprintf('FFNN  basis: %d correct: %d accuracy: %4f \n',i, accuracy, accuracy/length_yte*100);


%%%%%%%matlab classify%%%%%%%%
class = classify(h_te', h_tr', ytr);
y_hat_class = class == yte;
corrects = sum(y_hat_class);
fprintf('Classify basis: %d correct: %d accuracy: %4f \n',i, corrects, corrects/length_yte*100);


%%%%%%%%%multi layer nn%%%%%%%
ipconnect = zeros(2,44);
ipconnect(1,:) = 1;
ipconnect = boolean(ipconnect);

multinet = network;
multinet.numInputs = 44;
multinet.numLayers = 2;
multinet.biasConnect = [1;1];
multinet.layerConnect = [0 0;1 0];
multinet.inputConnect = ipconnect;
multinet.outputConnect = [0 1];
multinet.trainfcn = 'trainbfg';
[m_net,tr] = train(multinet, h_tr, ytr_t);

% get the target matrix 4 rows given one grouping vector
function[y_out] = get_target_matrix(y)
[n_trails, ~] = size(y);
y_1 = ones(n_trails,1); y_2= ones(n_trails,1); y_3=ones(n_trails,1); y_4 = ones(n_trails,1);
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
