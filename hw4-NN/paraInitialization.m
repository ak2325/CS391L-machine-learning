function [initial_nn_params, Nhlayers] = paraInitialization(input_layer_size,...
                                   layer1_size, ...
                                   layer2_size, ...
                                   layer3_size, ...
                                   layer4_size)
initial_nn_params = 0;
switch nargin
    case 3
        Nhlayers = 1;
        initial_Theta1 = randInitializeWeights(input_layer_size, layer1_size);
        initial_Theta2 = randInitializeWeights(layer1_size,layer2_size);
        initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
    case 4
        Nhlayers = 2;
        initial_Theta1 = randInitializeWeights(input_layer_size, layer1_size);
        initial_Theta2 = randInitializeWeights(layer1_size,layer2_size);
        initial_Theta3 = randInitializeWeights(layer2_size,layer3_size);
        initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];
    case 5
        Nhlayers = 3;
        initial_Theta1 = randInitializeWeights(input_layer_size, layer1_size);
        initial_Theta2 = randInitializeWeights(layer1_size,layer2_size);
        initial_Theta3 = randInitializeWeights(layer2_size,layer3_size);
        initial_Theta4 = randInitializeWeights(layer3_size,layer4_size);
        initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:) ; initial_Theta4(:)];                          
                               
end