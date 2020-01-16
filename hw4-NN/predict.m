function p = predict(X,actfunc,Theta1,Theta2,Theta3,Theta4)
m = size(X, 1);
p = zeros(size(X, 1), 1);
if actfunc=="sigmoid"
    actfunction = @(z) sigmoid(z);
    actfuncgrad = @(z) sigmoidGradient(z);
elseif actfunc=="tanh2"
    actfunction = @(z) tanh2(z);
    actfuncgrad = @(z) tanhGradient(z);
elseif actfunc=="ReLU"
    actfunction = @(z) ReLU(z);
    actfuncgrad = @(z) ReLUGradient(z);
elseif actfunc=="SeLU"
    actfunction = @(z) SeLU(z);
    actfuncgrad = @(z) SeLUGradient(z);
end

switch nargin
    case 4
h1 = actfunction([ones(m,1) X] * Theta1');
h2 = actfunction([ones(m,1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);
    case 5
h1 = actfunction([ones(m,1) X] * Theta1');
h2 = actfunction([ones(m,1) h1] * Theta2');
h3 = actfunction([ones(m,1) h2] * Theta3');
[dummy, p] = max(h3, [], 2);
    case 6
h1 = actfunction([ones(m,1) X] * Theta1');
h2 = actfunction([ones(m,1) h1] * Theta2');
h3 = actfunction([ones(m,1) h2] * Theta3');
h4 = actfunction([ones(m,1) h3] * Theta4');
[dummy, p] = max(h4,[],2);

end
