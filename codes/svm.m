rng(1);
lowerFr=2;
upperFr=30;

data = load_data('k3b',0.4, 1, 0);
xtr = data.Xtr;
ytr = data.Ytr;
xte = data.Xte;
yte = data.Yte;
M = 70;

k3b_acc = MultiSVMClassifier(xtr, ytr, xte, yte, lowerFr, upperFr, M);

data = load_data('k6b',0.4, 1, 0);
xtr = data.Xtr;
ytr = data.Ytr;
xte = data.Xte;
yte = data.Yte;
M = 28;

k6b_acc = MultiSVMClassifier(xtr, ytr, xte, yte, lowerFr, upperFr, M);

data = load_data('l1b',0.4, 1, 0);
xtr = data.Xtr;
ytr = data.Ytr;
xte = data.Xte;
yte = data.Yte;
M = 40;

l1b_acc = MultiSVMClassifier(xtr, ytr, xte, yte, lowerFr, upperFr, M);

disp(k3b_acc);
disp(k6b_acc);
disp(l1b_acc);

function [acc] = MultiSVMClassifier(xtr, ytr, xte, yte, lowerFr, upperFr, M)
    str = dostft_vector(xtr, 128, 64, 'hann', lowerFr, upperFr);
    ste = dostft_vector(xte, 128, 64, 'hann', lowerFr, upperFr);

    coefftr = pca(abs(str), 'NumComponents', M);
    coeffte = pca(abs(ste), 'NumComponents', M);

    ytepred = multisvm(coefftr,ytr,coeffte);

    acc = (nnz(ytepred == yte)/numel(yte))*100;
end