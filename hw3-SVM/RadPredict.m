function pred = RadPredict(model, X)

m = size(X, 1);
p = zeros(m, 1);
pred = zeros(m, 1);

X1 = sum(X.^2, 2);
X2 = sum(model.X.^2, 2)';
K = X1+X2-2*X*model.X';
K = gaussianKernel(1, 0, model.sigma) .^ K;
K = K.*model.y'.*model.alphas';
p = sum(K, 2) + model.b;

pred(p >= 0) =  1;
pred(p <  0) =  -1;

end