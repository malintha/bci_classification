rng(1); % For reproducability

lowerFr=2;
upperFr=30;

data = load_data('k3b',0.4, 1, 0);
xtr = data.Xtr;
ytr = data.Ytr;
yc = categorical(ytr);

xte = data.Xte;
yte = data.Yte;

%Mu = 3-7, lowerFr=6Hz upperFr=14Hz,  Mu+Beta = 3-15,lowerFr=6Hz upperFr=30Hz
%Only set numbers which are divisible by 2
str = abs(dostft(xtr, 128, 64, 'hann', lowerFr, upperFr));
ste = abs(dostft(xte, 128, 64, 'hann', lowerFr, upperFr));

M = 70;
% pc_tr = pca(str, 'NumComponents', M);
% pc_te = pca(ste, 'NumComponents', M);

mdl_tr = rica(str, 108);
mdl_te = rica(ste, 72);

pc_tr = transform(mdl_tr, str);
pc_te = transform(mdl_te, ste);