function g = ReLUGradient(z)

g = zeros(size(z));
g(z>0) = 1;

end