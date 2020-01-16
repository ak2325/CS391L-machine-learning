function [J,grad] = nnCostFunction(nn_params, X, y, lambda, costfunc, actfunc,...
                                   input_layer_size, ...
                                   layer1_size, ...
                                   layer2_size, ...
                                   layer3_size, ...
                                   layer4_size)
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
    case 11
s1 = layer1_size * (input_layer_size+1);
s2 = layer2_size * (layer1_size+1);
s3 = layer3_size * (layer2_size+1);
s4 = layer4_size * (layer3_size+1);
Theta1 = reshape(nn_params(1:s1),layer1_size,input_layer_size+1);
Theta2 = reshape(nn_params(s1+1:s1+s2),layer2_size,layer1_size+1);
Theta3 = reshape(nn_params(s1+s2+1:s1+s2+s3),layer3_size,layer2_size+1);
Theta4 = reshape(nn_params(s1+s2+s3+1:end),layer4_size,layer3_size+1);
m = size(X, 1);

X = [ones(m,1) X];
a1 = X;
z2 = a1*Theta1';
a2 = actfunction(z2);
a2 = [ones(m,1) a2];
z3 = a2*Theta2';
a3 = actfunction(z3);
a3 = [ones(m,1) a3];
z4 = a3*Theta3';
a4 = actfunction(z4);
a4 = [ones(m,1) a4];
z5 = a4*Theta4';
a5 = actfunction(z5);

ry = repmat(1:layer4_size, m, 1) == repmat(y, 1, layer4_size);

if costfunc=="crossentropy"
    cost = ry.*log(a5) + (1-ry).*log(1-a5);
    J = -sum(sum(cost,2))/m;
    regJ = lambda/(2*(s1+s2+s3+s4))*(sum(sum((Theta1(:,2:end)).^2,2))+sum(sum((Theta2(:,2:end)).^2,2))+...
    sum(sum((Theta3(:,2:end)).^2,2))+sum(sum((Theta4(:,2:end)).^2,2)));
    J = J+regJ;

    delta5 = a5-ry;
    delta4 = delta5*Theta4(:,2:end).*actfuncgrad(z4);
    delta3 = delta4*Theta3(:,2:end).*actfuncgrad(z3);
    delta2 = delta3*Theta2(:,2:end).*actfuncgrad(z2);
    Delta1 = delta2'*a1;
    Delta2 = delta3'*a2;
    Delta3 = delta4'*a3;
    Delta4 = delta5'*a4;
elseif costfunc=="hingeloss"
    cost = ry.*a5;
    J =1 -sum(sum(cost,2))/m;
    regJ = lambda/(2*(s1+s2+s3+s4))*(sum(sum((Theta1(:,2:end)).^2,2))+sum(sum((Theta2(:,2:end)).^2,2))+...
    sum(sum((Theta3(:,2:end)).^2,2))+sum(sum((Theta4(:,2:end)).^2,2)));
    J = J+regJ;

    delta5 = -ry.*sigmoid(z5).*(1-sigmoid(z5));
%     delta5 = a5-ry;
    delta4 = delta5*Theta4(:,2:end).*actfuncgrad(z4);
    delta3 = delta4*Theta3(:,2:end).*actfuncgrad(z3);
    delta2 = delta3*Theta2(:,2:end).*actfuncgrad(z2);
    Delta1 = delta2'*a1;
    Delta2 = delta3'*a2;
    Delta3 = delta4'*a3;
    Delta4 = delta5'*a4;
end
Theta1_grad = Delta1/m+lambda/m*[zeros(layer1_size,1) Theta1(:,2:end)];
Theta2_grad = Delta2/m+lambda/m*[zeros(layer2_size,1) Theta2(:,2:end)];
Theta3_grad = Delta3/m+lambda/m*[zeros(layer3_size,1) Theta3(:,2:end)];
Theta4_grad = Delta4/m+lambda/m*[zeros(layer4_size,1) Theta4(:,2:end)];
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:) ; Theta4_grad(:)];

    case 10
s1 = layer1_size * (input_layer_size+1);
s2 = layer2_size * (layer1_size+1);
s3 = layer3_size * (layer2_size+1);
Theta1 = reshape(nn_params(1:s1),layer1_size,input_layer_size+1);
Theta2 = reshape(nn_params(s1+1:s1+s2),layer2_size,layer1_size+1);
Theta3 = reshape(nn_params(s1+s2+1:end),layer3_size,layer2_size+1);
m = size(X, 1);

X = [ones(m,1) X];
a1 = X;
z2 = a1*Theta1';
a2 = actfunction(z2);
a2 = [ones(m,1) a2];
z3 = a2*Theta2';
a3 = actfunction(z3);
a3 = [ones(m,1) a3];
z4 = a3*Theta3';
a4 = sigmoid(z4);

ry = repmat(1:layer3_size, m, 1) == repmat(y, 1, layer3_size);

if costfunc=="crossentropy"
    cost = -ry.*log(a4) - (1-ry).*log(1-a4);
    J = sum(sum(cost,2))/m;
    regJ = lambda/(2*(s1+s2+s3))*(sum(sum((Theta1(:,2:end)).^2,2))+sum(sum((Theta2(:,2:end)).^2,2))+...
    sum(sum((Theta3(:,2:end)).^2,2)));
    J = J+regJ;

    delta4 = a4-ry;
    delta3 = delta4*Theta3(:,2:end).*actfuncgrad(z3);
    delta2 = delta3*Theta2(:,2:end).*actfuncgrad(z2);
    Delta1 = delta2'*a1;
    Delta2 = delta3'*a2;
    Delta3 = delta4'*a3;
elseif costfunc=="hingeloss"
    cost = ry.*a4;
    J = 1-sum(sum(cost,2))/m;
    regJ = lambda/(2*(s1+s2+s3))*(sum(sum((Theta1(:,2:end)).^2,2))+sum(sum((Theta2(:,2:end)).^2,2))+...
    sum(sum((Theta3(:,2:end)).^2,2)));
    J = J+regJ;

%     delta4 = -ry.*a4.*(1-a4);
    delta4 = a4-ry;
    delta3 = delta4*Theta3(:,2:end).*actfuncgrad(z3);
    delta2 = delta3*Theta2(:,2:end).*actfuncgrad(z2);
    Delta1 = delta2'*a1;
    Delta2 = delta3'*a2;
    Delta3 = delta4'*a3;
end
Theta1_grad = Delta1/m+lambda/m*[zeros(layer1_size,1) Theta1(:,2:end)];
Theta2_grad = Delta2/m+lambda/m*[zeros(layer2_size,1) Theta2(:,2:end)];
Theta3_grad = Delta3/m+lambda/m*[zeros(layer3_size,1) Theta3(:,2:end)];
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];

    case 9
s1 = layer1_size * (input_layer_size+1);
s2 = layer2_size * (layer1_size+1);
Theta1 = reshape(nn_params(1:layer1_size * (input_layer_size + 1)), ...
                 layer1_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (layer1_size * (input_layer_size + 1))):end), ...
                 layer2_size, (layer1_size + 1));
m = size(X, 1);

X = [ones(m,1) X];
a1 = X;
z2 = a1*Theta1';
a2 = actfunction(z2);
a2 = [ones(m,1) a2];
z3 = a2*Theta2';
a3 = actfunction(z3);

ry = repmat(1:layer2_size, m, 1) == repmat(y, 1, layer2_size);

if costfunc=="crossentropy"
    cost = ry.*log(a3) + (1-ry).*log(1-a3);
    J = -sum(sum(cost,2))/m;
    regJ = lambda/(2*(s1+s2))*(sum(sum((Theta1(:,2:end)).^2,2))+sum(sum((Theta2(:,2:end)).^2,2)));
    J = J+regJ;

    delta3 = a3-ry;
    delta2 = delta3*Theta2(:,2:end).*actfuncgrad(z2);
    Delta1 = delta2'*a1;
    Delta2 = delta3'*a2; 
elseif costfunc=="hingeloss"
    cost = ry.*a3;
    J = 1-sum(sum(cost,2))/m;
    regJ = lambda/(2*(s1+s2))*(sum(sum((Theta1(:,2:end)).^2,2))+sum(sum((Theta2(:,2:end)).^2,2)));
    J = J+regJ;

    delta3 = -ry.*sigmoid(z3).*(1-sigmoid(z3));
%     delta3 = a3-ry;
    delta2 = delta3*Theta2(:,2:end).*actfuncgrad(z2);
    Delta1 = delta2'*a1;
    Delta2 = delta3'*a2;
end
Theta1_grad = Delta1/m+lambda/m*[zeros(layer1_size,1) Theta1(:,2:end)];
Theta2_grad = Delta2/m+lambda/m*[zeros(layer2_size,1) Theta2(:,2:end)];
grad = [Theta1_grad(:) ; Theta2_grad(:)];
    
end
