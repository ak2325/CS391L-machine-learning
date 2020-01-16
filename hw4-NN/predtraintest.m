function [predtrain,predtest] = predtraintest(Xtrain,Xtest,actfunc,nn_params,...
    input_layer_size,layer1_size,layer2_size,layer3_size,layer4_size)

switch nargin
    case 7
    Theta1 = reshape(nn_params(1:layer1_size * (input_layer_size + 1)), ...
                 layer1_size, (input_layer_size + 1));
    Theta2 = reshape(nn_params((1 + (layer1_size * (input_layer_size + 1))):end), ...
                 layer2_size, (layer1_size + 1));
    predtrain = predict(Xtrain,actfunc,Theta1,Theta2);
    predtest = predict(Xtest,actfunc,Theta1,Theta2);
    case 8
    s1 = layer1_size * (input_layer_size+1);
    s2 = layer2_size * (layer1_size+1);
    Theta1 = reshape(nn_params(1:s1),layer1_size,input_layer_size+1);
    Theta2 = reshape(nn_params(s1+1:s1+s2),layer2_size,layer1_size+1);
    Theta3 = reshape(nn_params(s1+s2+1:end),layer3_size,layer2_size+1);
    predtrain = predict(Xtrain,actfunc,Theta1,Theta2,Theta3);
    predtest = predict(Xtest,actfunc,Theta1,Theta2,Theta3);
    case 9
    s1 = layer1_size * (input_layer_size+1);
    s2 = layer2_size * (layer1_size+1);
    s3 = layer3_size * (layer2_size+1);
    Theta1 = reshape(nn_params(1:s1),layer1_size,input_layer_size+1);
    Theta2 = reshape(nn_params(s1+1:s1+s2),layer2_size,layer1_size+1);
    Theta3 = reshape(nn_params(s1+s2+1:s1+s2+s3),layer3_size,layer2_size+1);
    Theta4 = reshape(nn_params(s1+s2+s3+1:end),layer4_size,layer3_size+1);
    predtrain = predict(Xtrain,actfunc,Theta1,Theta2,Theta3,Theta4);
    predtest = predict(Xtest,actfunc,Theta1,Theta2,Theta3,Theta4);

end