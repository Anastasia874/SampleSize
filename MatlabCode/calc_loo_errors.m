function [test_err, train_err, predictions] = calc_loo_errors(y, X, lambda, alpha)

m = length(y);
idx = 1:m;
test_err = zeros(m, 1);
train_err = zeros(m, 1);
predictions = zeros(m);
for i = 1:m
    idx_test = i;
    idx_train = idx(idx ~= idx_test);
    [wd, wg, ~, ~, sigma, P] = gdl_par(y(idx_train), X(idx_train, :), ...
                                                            lambda, alpha);
    w = [wd, wg];
    % to choose y we do not need to know the values of lambda and alpha
    [~, ~, p1] = logLamDensity(ones(m, 1), X, w, P, sigma, 0, 1);   
    [~, ~, p0] = logLamDensity(zeros(m, 1), X, w, P, sigma, 0, 1); 
    predictions(:, i) = p1 > p0;
    test_err(i) = mean(predictions(idx_test, i) ~= y(idx_test));
    train_err(i) = mean(predictions(idx_train, i) ~= y(idx_train));
end

predictions = mean(predictions, 2);
test_err = mean(test_err);
train_err = mean(train_err);

end