% load data
data = load_data('k3b',0.5, 1);
xtr = data.Xtr;
ytr = data.Ytr;

% do stft with 128 and 64 overlap
s = dostft(xtr, 128, 64);


