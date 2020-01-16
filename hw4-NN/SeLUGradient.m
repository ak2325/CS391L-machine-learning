function g = SeLUGradient(z)

lambda = 1.0507;
alpha = 1.6732;
g = zeros(size(z));
g(z>0) = lambda;
g(z<=0) = lambda*alpha.*exp(z(z<=0));

end