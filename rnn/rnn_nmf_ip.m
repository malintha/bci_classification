rng(1); % For reproducability

lowerFr=2;
upperFr=30;

data = load_data('k3b',0.4, 1, 0);
xtr = data.Xtr;
ytr = data.Ytr;
xte = data.Xte;
yte = data.Yte;
iterations_1=500;
iterations_2=500;
basis_vecs=70;

k3b_acc = NMF_RNN(xtr, ytr, xte, yte, basis_vecs, iterations_1, iterations_2, lowerFr, upperFr, 70);

data = load_data('k6b',0.6, 1, 0);
xtr = data.Xtr;
ytr = data.Ytr;
xte = data.Xte;
yte = data.Yte;
iterations_1=1000;
iterations_2=1000;
basis_vecs=40;

k6b_acc = NMF_RNN(xtr, ytr, xte, yte, basis_vecs, iterations_1, iterations_2, lowerFr, upperFr, 40);

data = load_data('l1b',0.4, 1, 0);
xtr = data.Xtr;
ytr = data.Ytr;
xte = data.Xte;
yte = data.Yte;
iterations_1=500;
iterations_2=500;
basis_vecs=70;

l1b_acc = NMF_RNN(xtr, ytr, xte, yte, basis_vecs, iterations_1, iterations_2, lowerFr, upperFr, 70);

disp(k3b_acc);
disp(k6b_acc);
disp(l1b_acc);

function [acc] = NMF_RNN(xtr, ytr, xte, yte, basis_vecs, iterations_1, iterations_2, lowerFr, upperFr, elem)
    [h_tr, h_te] = nmf(xtr, xte, basis_vecs, iterations_1, iterations_2, lowerFr, upperFr);

    tr_trails = size(xtr,3);
    sTr = h_tr';
    xc = cell(tr_trails,1); % Cell vector of 3 channels each with 1000 frames
    for i = 1:tr_trails
        xc(i) = {reshape(sTr(i,:),[elem, 1])};
    end

    inputSize = elem;
    outputSize = 150;
    outputMode = 'last';
    numClasses = 4;
    miniBatchSize = 50;

    layers = [ ...
        sequenceInputLayer(inputSize)
        lstmLayer(outputSize,'OutputMode',outputMode)
        lstmLayer(outputSize,'OutputMode',outputMode)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];

    maxEpochs = 1500;
    shuffle = 'never';

    options = trainingOptions('sgdm', ...
        'MaxEpochs',maxEpochs, ...
        'Shuffle', shuffle, ...
        'MiniBatchSize',miniBatchSize, ...
        'Plots','training-progress');

    yc = categorical(ytr);
    net = trainNetwork(xc,yc,layers,options);

    te_trails = size(xte,3);
    sTe = h_te';
    xtec = cell(te_trails,1);
    for i = 1:te_trails
        xtec(i) = {reshape(sTe(i,:),[elem, 1])};
    end

    Ytec = categorical(yte);
    YPred = classify(net,xtec);

    acc = (nnz(YPred == Ytec)/numel(Ytec))*100;
end