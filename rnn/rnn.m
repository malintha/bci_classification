rng(1); % For reproducability

% 1000 frames, 3 channels, 90 trials
data = load_data('k3b',0.5, 1, 0);
Xtr = data.Xtr;
Ytr = data.Ytr;
Xte = data.Xte;
Yte = data.Yte;

k3b_acc = RNN(Xtr, Ytr, Xte, Yte);

data = load_data('k6b',0.5, 1, 0);
Xtr = data.Xtr;
Ytr = data.Ytr;
Xte = data.Xte;
Yte = data.Yte;

k6b_acc = RNN(Xtr, Ytr, Xte, Yte);

data = load_data('l1b',0.5, 1, 0);
Xtr = data.Xtr;
Ytr = data.Ytr;
Xte = data.Xte;
Yte = data.Yte;

l1b_acc = RNN(Xtr, Ytr, Xte, Yte);

disp(k3b_acc);
disp(k6b_acc);
disp(l1b_acc);

function acc = RNN(Xtr, Ytr, Xte, Yte)
    tr_trails = size(Xtr,3);

    % Generate cell of matrices for each of the observations
    % Generate the time series objects
    X_t_c_f = permute(Xtr, [3 2 1]);
    Xc = cell(tr_trails,1); % Cell vector of 3 channels each with 1000 frames
    for i = 1:tr_trails
        Xc(i) = {reshape(X_t_c_f(i,:,:),[3,1000])};
    end

    % Create categorical Ytr array
    Yc = categorical(Ytr);

    inputSize = 3;
    outputSize = 10;
    outputMode = 'last';
    numClasses = 4;
    miniBatchSize = 50;

    layers = [ ...
        sequenceInputLayer(inputSize)
        lstmLayer(outputSize,'OutputMode',outputMode)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];

    maxEpochs = 1500;
    shuffle = 'never';

    options = trainingOptions('sgdm', ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ... 
        'Shuffle', shuffle, ...
        'Plots','training-progress');

    net = trainNetwork(Xc,Yc,layers,options);

    te_trials = size(Xte,3);

    Xte_t_c_f = permute(Xte, [3 2 1]);
    Xtec = cell(te_trials,1); % Cell vector of 3 channels each with 1000 frames
    for i = 1:te_trials
        Xtec(i) = {reshape(Xte_t_c_f(i,:,:),[3,1000])};
    end

    Ytec = categorical(Yte);

    YPred = classify(net,Xtec);

    acc = (nnz(YPred == Ytec)/numel(Ytec))*100;
end