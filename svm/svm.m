rng(1);
data = load_data('k3b',0.4, 1, 0);
xtr = data.Xtr;
ytr = data.Ytr;
xte = data.Xte;
yte = data.Yte;

%Mu = 3-7, lowerFr=6Hz upperFr=14Hz,  Mu+Beta = 3-15,lowerFr=6Hz upperFr=30Hz
%Only set numbers which are divisible by 2
lowerFr=2;
upperFr=30;
str = dostft(xtr, 128, 64, 'hann', lowerFr, upperFr);
ste = dostft(xte, 128, 64, 'hann', lowerFr, upperFr);

M = 70;
coefftr = pca(abs(str), 'NumComponents', M);
coeffte = pca(abs(ste), 'NumComponents', M);

ytepred = multisvm(coefftr,ytr,coeffte);

acc = (nnz(ytepred == yte)/numel(yte))*100;
disp(acc);