rng(1);
lowerFr=2;
upperFr=30;
iterations_1=500;
iterations_2=500;

data = load_data('k3b',0.4, 1, 0);
xtr = data.Xtr;
ytr = data.Ytr;
xte = data.Xte;
yte = data.Yte;
basis_vecs=70;

k3b_acc = NMF_SVM_Classifier(xtr, ytr, xte, yte,iterations_1, iterations_2, lowerFr, upperFr, basis_vecs);

data = load_data('k6b',0.6, 1, 0);
xtr = data.Xtr;
ytr = data.Ytr;
xte = data.Xte;
yte = data.Yte;
basis_vecs=30;
iterations_1=1000;
iterations_2=1000;

k6b_acc = NMF_SVM_Classifier(xtr, ytr, xte, yte,iterations_1, iterations_2, lowerFr, upperFr, basis_vecs);

data = load_data('l1b',0.4, 1, 0);
xtr = data.Xtr;
ytr = data.Ytr;
xte = data.Xte;
yte = data.Yte;
basis_vecs=40;
iterations_1=500;
iterations_2=500;

l1b_acc = NMF_SVM_Classifier(xtr, ytr, xte, yte,iterations_1, iterations_2, lowerFr, upperFr, basis_vecs);

disp(k3b_acc);
disp(k6b_acc);
disp(l1b_acc);

function [acc] = NMF_SVM_Classifier(xtr, ytr, xte, yte,iterations_1, iterations_2, lowerFr, upperFr, basis_vecs)
    [h_tr, h_te] = nmf(xtr, xte, basis_vecs, iterations_1, iterations_2, lowerFr, upperFr);
    ytepred = multisvm(h_tr',ytr,h_te');
    acc = (nnz(ytepred == yte)/numel(yte))*100;
end