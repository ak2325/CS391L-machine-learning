function g = tanhGradient(z)

g = (exp(z)+exp(-z))./(exp(z)+exp(-z))-(exp(z)-exp(-z)).*(exp(z)-exp(-z))./(exp(z)+exp(-z)).^2;

end