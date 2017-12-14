rng(1); % For reproducability

lowerFr=2;
upperFr=30;

data_k3b = load_data('k3b',0.4, 1, 0);
xtr = data_k3b.Xtr;
ytr = data_k3b.Ytr;
xte = data_k3b.Xte;
yte = data_k3b.Yte;
M = 70;

k3b_acc = PCA_NB_Classifier(xtr, ytr, xte, yte, lowerFr, upperFr, M);

data_k6b = load_data('k6b',0.4, 1, 0);
xtr = data_k6b.Xtr;
ytr = data_k6b.Ytr;
xte = data_k6b.Xte;
yte = data_k6b.Yte;
M = 28;

k6b_acc = PCA_NB_Classifier(xtr, ytr, xte, yte, lowerFr, upperFr, M);

data_l1b = load_data('l1b',0.4, 1, 0);
xtr = data_l1b.Xtr;
ytr = data_l1b.Ytr;
xte = data_l1b.Xte;
yte = data_l1b.Yte;
M = 40;

l1b_acc = PCA_NB_Classifier(xtr, ytr, xte, yte, lowerFr, upperFr, M);

disp(k3b_acc);
disp(k6b_acc);
disp(l1b_acc);

function [acc] = PCA_NB_Classifier(xtr, ytr, xte, yte, lowerFr, upperFr, M)
    str = abs(dostft_vector(xtr, 128, 64, 'hann', lowerFr, upperFr));
    ste = abs(dostft_vector(xte, 128, 64, 'hann', lowerFr, upperFr));
    
    pc_tr = pca(str, 'NumComponents', M);
    pc_te = pca(ste, 'NumComponents', M);

    model = fitcnb(pc_tr,ytr);
    ytepred = predict(model, pc_te);

    acc = (nnz(ytepred == yte)/numel(yte))*100;
end