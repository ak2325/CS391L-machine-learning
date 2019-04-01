function pred = PolyPredict(model, X)

m = size(X, 1);
pred = zeros(m, 1);

K = ((1/model.l)^2.*X*model.X'+1).^model.p;
p = K.*model.y'.*model.alphas';
p = sum(p, 2) + model.b;

pred(p >= 0) =  1;
pred(p <  0) =  -1;

end