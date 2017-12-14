rng(1);
lowerFr=2;
upperFr=30;

data_k3b = load_data('k3b',0.4, 1, 0);
xtr = data_k3b.Xtr;
ytr = data_k3b.Ytr;
xte = data_k3b.Xte;
yte = data_k3b.Yte;
M =70;

k3b_acc = NaiveBayesClassifier(xtr,xte,ytr,yte,lowerFr,upperFr,M);

data_k3b = load_data('k6b',0.4, 1, 0);
xtr = data_k3b.Xtr;
ytr = data_k3b.Ytr;
xte = data_k3b.Xte;
yte = data_k3b.Yte;
M = 28;

k6b_acc = NaiveBayesClassifier(xtr,xte,ytr,yte,lowerFr,upperFr,M);

data_k3b = load_data('l1b',0.4, 1, 0);
xtr = data_k3b.Xtr;
ytr = data_k3b.Ytr;
xte = data_k3b.Xte;
yte = data_k3b.Yte;
M = 40;

l1b_acc = NaiveBayesClassifier(xtr,xte,ytr,yte,lowerFr,upperFr,M);

disp(k3b_acc);
disp(k6b_acc);
disp(l1b_acc);

function [acc] = NaiveBayesClassifier(xtr,xte,ytr,yte,lowerFr,upperFr,M)
    str = dostft_vector(xtr, 128, 64, 'hann', lowerFr, upperFr);
    ste = dostft_vector(xte, 128, 64, 'hann', lowerFr, upperFr);

    coefftr = pca(abs(str), 'NumComponents', M);
    coeffte = pca(abs(ste), 'NumComponents', M);

    model = fitcnb(coefftr,ytr);
    ytepred = predict(model, coeffte);

    acc = (nnz(ytepred == yte)/numel(yte))*100;
end