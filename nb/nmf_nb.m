rng(1);
lowerFr=2;
upperFr=30;

data_k3b = load_data('k3b',0.4, 1, 0);
xtr = data_k3b.Xtr;
ytr = data_k3b.Ytr;
xte = data_k3b.Xte;
yte = data_k3b.Yte;
basis_vecs=70;
iterations_1=100;
iterations_2=100;

k3b_acc = NMF_NB_Classifier(xtr, ytr, xte, yte, basis_vecs, iterations_1, iterations_2, lowerFr, upperFr);

data_k6b = load_data('k6b',0.6, 1, 0);
xtr = data_k6b.Xtr;
ytr = data_k6b.Ytr;
xte = data_k6b.Xte;
yte = data_k6b.Yte;
basis_vecs=30;
iterations_1=1000;
iterations_2=1000;

k6b_acc = NMF_NB_Classifier(xtr, ytr, xte, yte, basis_vecs, iterations_1, iterations_2, lowerFr, upperFr);

data_l1b = load_data('l1b',0.4, 1, 0);
xtr = data_l1b.Xtr;
ytr = data_l1b.Ytr;
xte = data_l1b.Xte;
yte = data_l1b.Yte;
basis_vecs=40;
iterations_1=1000;
iterations_2=1000;

l1b_acc = NMF_NB_Classifier(xtr, ytr, xte, yte, basis_vecs, iterations_1, iterations_2, lowerFr, upperFr);

disp(k3b_acc);
disp(k6b_acc);
disp(l1b_acc);

function [acc] = NMF_NB_Classifier(xtr, ytr, xte, yte, basis_vecs, iterations_1, iterations_2, lowerFr, upperFr)
    [h_tr, h_te] = nmf(xtr, xte, basis_vecs, iterations_1, iterations_2, lowerFr, upperFr);
    model = fitcnb(h_tr',ytr);
    ytepred = predict(model, h_te');

    acc = (nnz(ytepred == yte)/numel(yte))*100;
end