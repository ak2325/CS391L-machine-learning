%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CS 391L
% hw3
% Yue WU, yw9998
% 03/18/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all;

% ntrain = [100,500,1000,2000,5000];
ntrain = 5000;

% 4 vs 9
load results3.mat

% apply optimal C for both Polynomial Kernel and Gaussian Kernel
fprintf('Calculating accuracy for different p and different l...\n')
lcandi = [1,3,5,10,30,50,100,300];
pcandi = [1,3,5,7,9,11,13,15,17,19];
accupl = comparep(A1,A1Labels,Cp,lcandi,pcandi,B1,B1Labels);

fprintf('Calculating accuracy for different sigma ...\n')
sigcandi = [0.01,0.03,0.1,0.3,1,3,10,30];
accur = comparesig(A1,A1Labels,Cr,sigcandi,B1,B1Labels);
