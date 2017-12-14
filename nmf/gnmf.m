% Group NMF Implementation. Here we use a vectorized STFT tranformation.
rng(1);
clear;
data_k3b = load_data('k3b',0.6, 1, 0);
data_k6b = load_data('k6b',0.6, 1, 0);
data_l1b = load_data('l1b',0.6, 1, 0);

str_k3 = abs(dostft_vector(data_k3b.Xtr, 128, 64, 'hann', 2, 30));
str_k6 = abs(dostft_vector(data_k6b.Xtr, 128, 64, 'hann', 2, 30));
str_l1 = abs(dostft_vector(data_l1b.Xtr, 128, 64, 'hann', 2, 30));

ste_k3 = abs(dostft_vector(data_k3b.Xte, 128, 64, 'hann', 2, 30));
ste_k6 = abs(dostft_vector(data_k6b.Xte, 128, 64, 'hann', 2, 30));
ste_l1 = abs(dostft_vector(data_l1b.Xte, 128, 64, 'hann', 2, 30));

% for common_b=4:50
%     for ind_b=1:10
% apply group NMF approach for all 3 training datasets
common_b = 4;
ind_b = 1;
for beta = 0.1:0.5:10
    for alpha = 0.1:0.5:10
% alpha = 1.05;
% beta = 0.5;
lambda = 0.5;
gamma = 0.5;
[a1, a2, a3, str1, str2, str3] = learn_basis_vectors(str_k3, str_k6, str_l1, common_b, ind_b, 5000, alpha, beta, gamma, lambda);

% learn weights from basis vectors using the test data
ste1 = learn_weights(ste_k3, a1, 500);
ste2 = learn_weights(ste_k6, a2, 500);
ste3 = learn_weights(ste_l1, a3, 500);

class_1 = classify(ste1(1:common_b,:)' ,str1(1:common_b,:)', data_k3b.Ytr);
class_2 = classify(ste2(1:common_b,:)' ,str2(1:common_b,:)', data_k6b.Ytr);
class_3 = classify(ste3(1:common_b,:)' ,str3(1:common_b,:)', data_l1b.Ytr);

y_hat_class_1 = class_1 == data_k3b.Yte;
y_hat_class_2 = class_2 == data_k6b.Yte;
y_hat_class_3 = class_3 == data_l1b.Yte;
corrects_classify_1 = sum(y_hat_class_1);
corrects_classify_2 = sum(y_hat_class_2);
corrects_classify_3 = sum(y_hat_class_3);

accuracy_k3b = corrects_classify_1/length(data_k3b.Yte)*100;
accuracy_k6b = corrects_classify_2/length(data_k6b.Yte)*100;
accuracy_l1b = corrects_classify_3/length(data_l1b.Yte)*100;
avg = (accuracy_k3b+accuracy_k6b+accuracy_l1b)/3;
fprintf('alpha: %4f beta: %4f common_b: %d ind_b %d k3b: %4f k6b: %4f l1b: %4f avg: %4f \n',alpha, beta, common_b, ind_b, accuracy_k3b, accuracy_k6b, accuracy_l1b, avg);

    end
end

% end
%%%%%%%single layer nn%%%%%%%%%
[acc1, acc2] = sl_nn(str1(1:common_b,:), data_k3b.Ytr, ste1(1:common_b,:), data_k3b.Yte, 0.002, 20000, 35);
fprintf('Basis: %d Accuracy of the classification using train data: %f \n',i,acc1);
fprintf('Basis: %d Accuracy of the classification using test data: %f \n',i,acc2);

function[s] = learn_weights(ste, a, iterations)
[~, c_te] = size(ste);
[~, c_a] = size(a);
s = unifrnd(0, 1, c_a, c_te);
for i=0:iterations
    s = s .* ((a'*ste)./(a'*a*s));
end
end

function[a1, a2, a3, s1, s2, s3] = learn_basis_vectors(tr_1, tr_2, tr_3, common_b, ind_b, iterations, alpha, beta, gamma, lambda)
[r_tr1, c_tr1] = size(tr_1);
[r_tr2, c_tr2] = size(tr_2);
[r_tr3, c_tr3] = size(tr_3);
s1 = unifrnd(0,1, common_b+ind_b, c_tr1);
s2 = unifrnd(0,1, common_b+ind_b, c_tr2);
s3 = unifrnd(0,1, common_b+ind_b, c_tr3);

a1 = unifrnd(0,1, r_tr1,common_b+ind_b);
a2 = unifrnd(0,1, r_tr2,common_b+ind_b);
a3 = unifrnd(0,1, r_tr3,common_b+ind_b);

for i=0:iterations
    s1 = s1 .* ((a1'*tr_1)./(a1'*a1*s1));
    s2 = s2 .* ((a2'*tr_2)./(a2'*a2*s2));
    s3 = s3 .* ((a3'*tr_3)./(a3'*a3*s3));
    
    J_1c = get_J_c(tr_1, s1, a1, a2, a3, common_b, alpha, lambda, gamma);
    J_1i = get_J_i(tr_1, s1, a1, a2, a3, common_b, ind_b, beta, lambda, gamma);
    
    a1(:,1:common_b) = a1(:,1:common_b).*J_1c;
    a1(:,common_b+1:common_b+ind_b) = a1(:,common_b+1:common_b+ind_b).*J_1i;
        
    J_2c = get_J_c(tr_2, s2, a2, a1, a3, common_b, alpha, lambda, gamma);
    J_2i = get_J_i(tr_2, s2, a2, a1, a3, common_b, ind_b, beta, lambda, gamma);
    
    a2(:,1:common_b) = a2(:,1:common_b).*J_2c;
    a2(:,common_b+1:common_b+ind_b) = a2(:,common_b+1:ind_b+common_b).*J_2i;
    
    J_3c = get_J_c(tr_3, s3, a3, a1, a1, common_b, alpha, lambda, gamma);
    J_3i = get_J_i(tr_3, s3, a3, a1, a1, common_b, ind_b, beta, lambda, gamma);
    
    a3(:,1:common_b) = a3(:,1:common_b).*J_3c;
    a3(:,common_b+1:common_b+ind_b) = a3(:,common_b+1:common_b+ind_b).*J_3i;
end
end

function[J_c] = get_J_c(tr, s, a1, a2, a3, common_b, alpha, lambda, gamma) 
    J_c_nume = tr*s(1:common_b,:)' + (alpha/lambda)*(a2(:,1:common_b) + a3(:,1:common_b));
    J_c_denom = (a1*s)*s(1:common_b,:)' + (alpha/lambda)*2*a1(:,1:common_b) + (gamma/lambda)*a1(:,1:common_b);
    J_c = J_c_nume./J_c_denom;
end

function[J_i] = get_J_i(tr, s, a1, a2, a3, common_b, ind_b, beta, lambda, gamma) 
    J_i_nume = tr*s(common_b+1:common_b+ind_b,:)' + (beta/lambda)*2*a1(:,common_b+1:common_b+ind_b);
    J_i_denom = (a1*s)*s(common_b+1:common_b+ind_b,:)' + (beta/lambda)*(a2(:,common_b+1:common_b+ind_b) + a3(:,common_b+1:common_b+ind_b))  + (gamma/lambda)*a1(:,common_b+1:common_b+ind_b);
    J_i = J_i_nume./J_i_denom;
end

function[new_y] = get_grouped_target_vector(y, target_len)
len_y = length(y);
step = target_len/len_y;
new_y = [];
for i=step:step:target_len
    temp_y = zeros(step,1);
    temp_y(:,1) = y(i/step);
    new_y = [new_y;temp_y];
end
end