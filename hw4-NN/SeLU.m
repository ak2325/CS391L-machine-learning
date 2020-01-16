function g = SeLU(z)

lambda = 1.0507;
alpha = 1.6732;
g = zeros(size(z));
g(z>0) = lambda*z(z>0);
g(z<=0) = lambda*alpha.*(exp(z(z<=0))-1);

end

