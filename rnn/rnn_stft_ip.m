rng(1); % For reproducability

lowerFr=2;
upperFr=30;

data = load_data('k3b',0.4, 1, 0);
xtr = data.Xtr;
ytr = data.Ytr;
xte = data.Xte;
yte = data.Yte;

k3b_acc = STFT_RNN(xtr, ytr, xte, yte, lowerFr, upperFr, 630);

data = load_data('k6b',0.6, 1, 0);
xtr = data.Xtr;
ytr = data.Ytr;
xte = data.Xte;
yte = data.Yte;

k6b_acc = STFT_RNN(xtr, ytr, xte, yte, lowerFr, upperFr, 630);

data = load_data('l1b',0.4, 1, 0);
xtr = data.Xtr;
ytr = data.Ytr;
xte = data.Xte;
yte = data.Yte;

l1b_acc = STFT_RNN(xtr, ytr, xte, yte, lowerFr, upperFr, 630);

disp(k3b_acc);
disp(k6b_acc);
disp(l1b_acc);

function [acc] = STFT_RNN(xtr, ytr, xte, yte, lowerFr, upperFr, elem)
    str = abs(dostft_vector(xtr, 128, 64, 'hann', lowerFr, upperFr));
    ste = abs(dostft_vector(xte, 128, 64, 'hann', lowerFr, upperFr));

    tr_trails = size(xtr,3);
    sTr = str';
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
    sTe = ste';
    xtec = cell(te_trails,1);
    for i = 1:te_trails
        xtec(i) = {reshape(sTe(i,:),[elem, 1])};
    end

    Ytec = categorical(yte);
    YPred = classify(net,xtec);

    acc = (nnz(YPred == Ytec)/numel(Ytec))*100;
end