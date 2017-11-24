% [h_tr, h_te] =  nmf('k3b', 70, 800, 500)

y_target = get_target_matrix(ytr);

% class = classify(h_te', h', yte);
net = feedforwardnet(4);
net = train(net,h_tr, y_target');
y = net(h_te);
% get accuracy
% normalize y
y_hat = y./max(y);
y_hat = y_hat == 1;

y_test = get_target_matrix(yte);
y_accuracy_test = y_hat'.*y_test;
accuracy = sum(sum(y_accuracy_test));
fprintf('correct: %d accuracy: %4f \n', accuracy, accuracy/95*100);


% get the target matrix 4 rows given one grouping vector
function[y_out] = get_target_matrix(y)
[n_trails, ~] = size(y);
y_1 = ones(n_trails,1); y_2= ones(n_trails,1); y_3=ones(n_trails,1); y_4 = ones(n_trails,1);
y_1(:,1) = 1;
y_2(:,1) = 2;
y_3(:,1) = 3;
y_4(:,1) = 4;
y_1 = y_1==y;
y_2 = y_2==y;
y_3 = y_3==y;
y_4 = y_4==y;
y_out = horzcat(y_1, y_2, y_3, y_4);
end
