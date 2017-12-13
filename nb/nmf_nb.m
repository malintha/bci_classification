rng(1);
lowerFr=2;
upperFr=30;
iterations_1=500;
iterations_2=500;
basis_vecs=70;

data = load_data('k3b',0.4, 1, 0);
xtr = data.Xtr;
ytr = data.Ytr;
xte = data.Xte;
yte = data.Yte;

[h_tr, h_te] = nmf(xtr, xte, basis_vecs, iterations_1, iterations_2, lowerFr, upperFr);
model = fitcnb(h_tr',ytr);
ytepred = predict(model, h_te');

acc = (nnz(ytepred == yte)/numel(yte))*100;
disp(acc);