clear all; close all;

load('digits.mat');
trainset = reshape(im2double(trainImages),[],60000);
testset = reshape(im2double(testImages),[],10000);

% ntrain = [100,500,1000,2000,5000];
ntrain = 100;
A = trainset(:,1:ntrain)';
A = A-mean(A,1);
ALabels = double(trainLabels(1:ntrain))';
% imshow(reshape(A(:,1),28,28))
B = testset';
B = B-mean(B,1);
BLabels = double(testLabels)';

% 0 vs rest
A1Labels = ALabels;
A1Labels(ALabels~=0) = -1;
A1Labels(ALabels==0) = 1;
B1Labels = BLabels;
B1Labels(BLabels~=0) = -1;
B1Labels(BLabels==0) = 1;

Cp = 0.01;
p = 1;
Cr = 1;
sigma = 10;

pmodel = PolyTrain(A, A1Labels, Cp, p, [], 20);
pred = PolyPredict(pmodel,B);
accup = sum(pred==B1Labels)/length(B1Labels);
rmodel = RadTrain(A, A1Labels, Cr, sigma, [], 20);
pred = RadPredict(rmodel,B);
accur = sum(pred==B1Labels)/length(B1Labels);