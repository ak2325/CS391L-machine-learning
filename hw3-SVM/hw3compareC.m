%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CS 391L
% hw3
% Yue WU, yw9998
% 03/17/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Polykernel: changes in C won't affect result much, notice l and p
% Gaussian: hard margin performs better

clear all; close all;

load('digits.mat');
trainset = reshape(im2double(trainImages),[],60000);
testset = reshape(im2double(testImages),[],10000);

% ntrain = [100,500,1000,2000,5000];
ntrain = 2000;
A = trainset(:,1:ntrain)';
A = A-mean(A,1);
ALabels = double(trainLabels(1:ntrain))';
% imshow(reshape(A(:,1),28,28))
B = testset';
B = B-mean(B,1);
BLabels = double(testLabels)';

p = 11;
l = 10;
sigma = 10;

%% 0 vs rest
A1Labels = ALabels;
A1Labels(ALabels~=0) = -1;
A1Labels(ALabels==0) = 1;
A1 = A;
B1Labels = BLabels;
B1Labels(BLabels~=0) = -1;
B1Labels(BLabels==0) = 1;
B1 = B;

% Polynomial SVM
[Cp,Cpcandi,perrorp] = PolyParams(A1,A1Labels, p, l);

% Radial Basis SVM
[Cr,Crcandi,perrorr] = RadParams(A1,A1Labels, sigma);

save results1.mat A1 A1Labels B1 B1Labels Cp Cpcandi perrorp ...
    Cr Crcandi perrorr

%% 7 vs rest
A1Labels = ALabels;
A1Labels(ALabels~=7) = -1;
A1Labels(ALabels==7) = 1;
A1 = A;
B1Labels = BLabels;
B1Labels(BLabels~=7) = -1;
B1Labels(BLabels==7) = 1;
B1 = B;

% Polynomial SVM
[Cp,Cpcandi,perrorp] = PolyParams(A1,A1Labels, p, l);

% Radial Basis SVM
[Cr,Crcandi,perrorr] = RadParams(A1,A1Labels, sigma);

save results2.mat A1 A1Labels B1 B1Labels Cp Cpcandi perrorp ...
    Cr Crcandi perrorr

%% 4 vs 9
idx = find(ALabels==4 | ALabels==9);
A1Labels = ALabels(idx);
A1 = A(idx,:);
A1Labels(A1Labels==9) = -1;
A1Labels(A1Labels==4) = 1;
idx = find(BLabels==4 | BLabels==9);
B1Labels = BLabels(idx);
B1 = B(idx,:);
B1Labels(B1Labels==9) = -1;
B1Labels(B1Labels==4) = 1;

% Polynomial SVM
[Cp,Cpcandi,perrorp] = PolyParams(A1,A1Labels, p, l);

% Radial Basis SVM
[Cr,Crcandi,perrorr] = RadParams(A1,A1Labels, sigma);

save results3.mat A1 A1Labels B1 B1Labels Cp Cpcandi perrorp ...
    Cr Crcandi perrorr

%% 0 vs 8
idx = find(ALabels==0 | ALabels==8);
A1Labels = ALabels(idx);
A1 = A(idx,:);
A1Labels(A1Labels==8) = -1;
A1Labels(A1Labels==0) = 1;
idx = find(BLabels==0 | BLabels==8);
B1Labels = BLabels(idx);
B1 = B(idx,:);
B1Labels(B1Labels==8) = -1;
B1Labels(B1Labels==0) = 1;

% Polynomial SVM
[Cp,Cpcandi,perrorp] = PolyParams(A1,A1Labels, p, l);

% Radial Basis SVM
[Cr,Crcandi,perrorr] = RadParams(A1,A1Labels, sigma);

save results4.mat A1 A1Labels B1 B1Labels Cp Cpcandi perrorp ...
    Cr Crcandi perrorr

%% (0,8,3) vs (1,7,9)
idx = find(ALabels==0 | ALabels==8 | ALabels==3 | ALabels==1 | ALabels==7 | ALabels==9);
A1Labels = ALabels(idx);
A1 = A(idx,:);
A1Labels(A1Labels==1 | A1Labels==7 | A1Labels==9) = -1;
A1Labels(A1Labels==0 | A1Labels==8 | A1Labels==3) = 1;
idx = find(BLabels==0 | BLabels==8 | BLabels==3 | BLabels==1 | BLabels==7 | BLabels==9);
B1Labels = BLabels(idx);
B1 = B(idx,:);
B1Labels(B1Labels==1 | B1Labels==7 | B1Labels==9) = -1;
B1Labels(B1Labels==0 | B1Labels==8 | B1Labels==3) = 1;

% Polynomial SVM
[Cp,Cpcandi,perrorp] = PolyParams(A1,A1Labels, p, l);

% Radial Basis SVM
[Cr,Crcandi,perrorr] = RadParams(A1,A1Labels, sigma);

save results5.mat A1 A1Labels B1 B1Labels Cp Cpcandi perrorp ...
    Cr Crcandi perrorr
