function[h_tr, h_te] = nmf(xtr, xte, basis_vecs, iterations_1, iterations_2, lower, upper)   
    rng(1);

    % do stft with 128 and 64 overlap
    str = dostft(xtr, 128, 64, 'hann', lower, upper);
    ste = dostft(xte, 128, 64, 'hann', lower, upper);

%     str = [str(5:15,:);str(25:45,:);str(55:75,:)];
%     ste = [ste(5:15,:);ste(25:45,:);ste(55:75,:)];
%     [sw, h_tr] = learnNMF(abs(s), iterations_1, basis_vecs);
    [w_tr, h_tr] = nnmf(abs(str), basis_vecs, 'algorithm', 'mult');
    h_te = learnNMF_H(abs(ste), w_tr, iterations_2);

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
%         w_denom = ones_ft*h';
%         w_new = w.*(w_nume./(w_denom + eps));

        %other approach
        s_nume = sum(w_nume,2);
        w_new = w.*s_nume;
        w_new = w_new ./ sum(w_new);

        %update rule for h
        h_nume = w'*x_over_wh;
%         h_denom = w'*ones_ft;
%         h_new = h.*(h_nume./(h_denom + eps));
        % other approach
        s_h_nume = sum(h_nume);
        h_new = h.* s_h_nume;

        %calculate the error
        x_hat = w*h;
        [x_rows, x_cols] = size(x);
        epsilon = sum(sum((x .* log(x./(x_hat + eps))) - x + x_hat,'omitnan'));
%         fprintf(' Iteration: %d Error: %4f\n',it, epsilon);
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
        epsilon = sum(sum((x .* log(x./(y_hat + eps))) - x + y_hat,'omitnan'));
%         fprintf(' Iteration: %d Error: %4f\n',it, epsilon);
        h = h_new;
    end
    end

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
end