function[h_tr, h_te] = plsi(xtr, xte, basis_vecs, iterations_1, iterations_2, lower, upper)   
    rng(1);

    % do stft with 128 and 64 overlap
    str = dostft_vector(xtr, 128, 64, 'blackman', lower, upper);
    ste = dostft_vector(xte, 128, 64, 'blackman', lower, upper);

    [w_tr, h_tr] = learnplsi(abs(str), iterations_1, basis_vecs);
    h_te = learnplsi_theta(abs(ste), w_tr, iterations_2);

    function[b,theta] = learnplsi(x, iterations, k)
    [rows, cols] = size(x);
    b = unifrnd(0,500,rows, k);
    theta = unifrnd(0,500,k, cols);
    ones_vv = ones(rows,rows);
    ones_kk = ones(k, k);

    for it=1:iterations
        b = b.* ((x./(b*theta + eps))*theta');
        b = b./(ones_vv*b + eps);

        %update rule for theta
        theta = theta .* (b'*(x./(b*theta + eps)));
        theta = theta./(ones_kk*theta + eps);

        %calculate the error
        x_hat = b*theta;
        epsilon = sum(sum((x .* log(x./(x_hat + eps))) - x + x_hat,'omitnan'));
%         fprintf(' Iteration: %d Error: %4f\n',it, epsilon);
    end
    end

    function[theta] = learnplsi_theta(x, b, iterations) 
    [w_rows, w_cols] = size(b);
    [~, x_cols] = size(x);

    theta = unifrnd(0,500,w_cols, x_cols);
    ones_kk = ones(w_cols,w_cols);
    for it=1:iterations
        %update rule for theta
        theta = theta .* (b'*(x./(b*theta + eps)));
        theta = theta./((ones_kk*theta + eps));
        %calculate the error
        y_hat = b*theta;
        epsilon = sum(sum((x .* log(x./(y_hat + eps))) - x + y_hat,'omitnan'));
%         fprintf(' Iteration: %d Error: %4f\n',it, epsilon);
    end
    end

end