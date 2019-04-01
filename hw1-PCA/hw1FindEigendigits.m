% A: x-k, training set
% m: 1-x, mean column vector
% V: x-k, k eigenvectors of the covariance matrix of A

function [m,V] = hw1FindEigendigits(A)
[x,k] = size(A);
m = mean(A,2);
if x<=k
    C = 1/k.*(A-m)*(A-m)';
    [V,D] = eig(C,'vector');
else
    C = 1/k.*(A-m)'*(A-m);
    [V,D] = eig(C,'vector');
    V = A*V;
end
end