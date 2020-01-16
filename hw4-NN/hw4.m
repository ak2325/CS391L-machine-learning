%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CS 391L
% hw4
% Yue WU, yw9998
% 04/18/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization
clear ; close all; clc

fprintf('Loading Data ...\n')
load digits.mat
trainset = reshape(im2double(trainImages),[],60000)';
testset = reshape(im2double(testImages),[],10000)';
trainLabels = trainLabels';
trainLabels(trainLabels==0) = 10;
testLabels = testLabels';
testLabels(testLabels==0) = 10;

[train1,train2] = size(trainset);
[test1,test2] = size(testset);
hidden_layer1_size = 25;    %
% hidden_layer2_size = 30;    %
% hidden_layer3_size = 4;    % change when # of hidden layers changes
num_labels = 10;  

% Visualization
% sel = randperm(train1);
% sel = sel(1:100);
% displayData(trainset(sel, :));

fprintf('\nInitializing Neural Network Parameters ...\n')
[nn_params, Nhlayers] = paraInitialization(train2,hidden_layer1_size,...
    num_labels); % change when # of hidden layers changes

%% Backpropagation
fprintf('\nTraining Neural Network... \n')
tic

lambda = 0;
alpha = 0.1;
Ntrain = 1000;
Nbatch = 200;
costfunc = "crossentropy";  % costfunc: 'crossentropy','hingeloss'
actfunc = "sigmoid";    % actfunc: 'sigmoid','tanh2','ReLU','SeLU'
thresh = 0.1;

i = 0;
sel = randperm(train1);
sel = sel(1:Ntrain);
X = trainset(sel,:);
y = trainLabels(sel,:);
cost = inf;
while cost>thresh
    avgcost = 0;
    i = i+1;
    for j = 1:round(Ntrain/Nbatch)
        sel = randperm(Ntrain);
        sel = sel(1:Nbatch);
        x = X(sel,:);
        yy = y(sel,:);
        [batchcost,grad] = nnCostFunction(nn_params, x, yy,lambda, costfunc, actfunc,...
            train2,hidden_layer1_size,num_labels); %
        avgcost = avgcost+batchcost;
        nn_params = nn_params-alpha*grad;
    end
    cost = avgcost/round(Ntrain/Nbatch);
    fprintf('%s %s| %s %4i | Cost: %4.6e\r',costfunc,actfunc,'iteration',i, cost);
end
toc
% Visualization
% displayData(Theta1(:, 2:end));

%% Prediction
[predtrain,predtest] = predtraintest(X,testset,actfunc,nn_params,...
    train2,hidden_layer1_size,num_labels); %
fprintf('\nTraining Set Accuracy: %f\n', mean(double(predtrain == y)) * 100);
fprintf('\nTesting Set Accuracy: %f\n', mean(double(predtest == testLabels)) * 100);
