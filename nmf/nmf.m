% load data
% data = load_data('k3b',0.5, 1);
% xtr = data.Xtr;
% ytr = data.Ytr;
xte = data.Xte;
yte = data.Yte;

% do stft with 128 and 64 overlap
s = dostft(xtr, 128, 64);
ste = abs(dostft(xte, 128, 64));
% s = s';
% separate the frames
% get the frames of class 1, 2, 3, 4
% s_1 = abs(get_frames_of_class(s, ytr, 1)');
% s_2 = abs(get_frames_of_class(s, ytr, 2)');
% s_3 = abs(get_frames_of_class(s, ytr, 3)');
% s_4 = abs(get_frames_of_class(s, ytr, 4)');
% 
% [sw_1, h1] = learnNMF(s_1, 800, 100);
% [sw_2, h2] = learnNMF(s_2, 800, 100);
% [sw_3, h3] = learnNMF(s_3, 800, 100);
% [sw_4, h4] = learnNMF(s_4, 800, 100);

% [sw, h] = learnNMF(abs(s), 800, 100);
% sw = [sw_1 sw_2 sw_3 sw_4];
h_te = learnNMF_H(ste, sw, 500);

% grouping_v = ones(400,1);
% grouping_v(101:200,:) = 2;
% grouping_v(201:300,:) = 3;
% grouping_v(301:400,:) = 4;

% ste_classify = ste';
% sw_classify = sw';
class = classify(h_te', h', yte);


% get the frames of a given class from a given spectrogram
function[s_out] = get_frames_of_class(s, y, c)
[rows, ~] = size(y);
s_out = [];
for i=1:rows
    if (y(i,1) == c)
        s_out = [s_out; s(i,:)];
    end
end
end

function[w,h] = learnNMF(x, iterations, b)
[rows, cols] = size(x);
w = rand(rows, b);
h = rand(b, cols);
ones_ft = ones(rows, cols);

for it=1:iterations
    %update rule for w
    wh = w*h;
    x_over_wh = x./wh;
    w_nume = x_over_wh*h';
    w_denom = ones_ft*h';
    w_new = w.*(w_nume./(w_denom + eps));
    
    %update rule for h
    h_nume = w'*x_over_wh;
    h_denom = w'*ones_ft;
    h_new = h.*(h_nume./(h_denom + eps));
    
    %calculate the error
    x_hat = w*h;
    [x_rows, x_cols] = size(x);
    epsilon = 0;
    for i=1:x_rows
        for j=1:x_cols
            epsilon = epsilon + x(i,j) * log(x(i,j)/x_hat(i,j)) - x(i,j) + x_hat(i,j);
        end
    end
    fprintf(' Iteration: %d Error: %4f\n',it, epsilon);
    w = w_new;
    h = h_new;
end
end


function[h] = learnNMF_H(x, w, iterations) 
[w_rows, w_cols] = size(w);
[~, x_cols] = size(x);

h = rand(w_cols, x_cols);
ones_ft = ones(w_rows,x_cols);
for it=1:iterations
    %update rule for w
    wh = w*h;
    y_over_wh = x./wh;
   
    %update rule for h
    h_nume = w'*y_over_wh;
    h_denom = w'*ones_ft;
    h_new = h.*(h_nume./(h_denom + eps));
    
    %calculate the error
    y_hat = w*h;
    [y_rows, y_cols] = size(x);
    epsilon = 0;
    for i=1:y_rows
        for j=1:y_cols
            epsilon = epsilon + x(i,j) * log(x(i,j)/y_hat(i,j)) - x(i,j) + y_hat(i,j);
        end
    end
    fprintf(' Iteration: %d Error: %4f\n',it, epsilon);
    h = h_new;
end
end