%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CS 391L
% hw3
% Yue WU, yw9998
% 03/17/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all;

% ntrain = [100,500,1000,2000,5000];
ntrain = 5000;

p = 11;
l = 10;
sigma = 10;

accu = zeros(5,6);

%% 0 vs rest
load results1.mat
[accup, accuph, accups, accur, accurs, accurh] = compareC(A1, ...
    A1Labels, Cp, p, l, Cr, sigma, B1, B1Labels);
accu(1,:) = [accup;accups;accuph;accur;accurs;accurh];

%% 7 vs rest
load results2.mat
[accup, accuph, accups, accur, accurs, accurh] = compareC(A1, ...
    A1Labels, Cp, p, l, Cr, sigma, B1, B1Labels);
accu(2,:) = [accup;accups;accuph;accur;accurs;accurh];

%% 4 vs 9
load results3.mat
[accup, accuph, accups, accur, accurs, accurh] = compareC(A1, ...
    A1Labels, Cp, p, l, Cr, sigma, B1, B1Labels);
accu(3,:) = [accup;accups;accuph;accur;accurs;accurh];

%% 0 vs 8
load results4.mat
[accup, accuph, accups, accur, accurs, accurh] = compareC(A1, ...
    A1Labels, Cp, p, l, Cr, sigma, B1, B1Labels);
accu(4,:) = [accup;accups;accuph;accur;accurs;accurh];

%% (0,8,3) vs (1,7,9)
load results5.mat
[accup, accuph, accups, accur, accurs, accurh] = compareC(A1, ...
    A1Labels, Cp, p, l, Cr, sigma, B1, B1Labels);
accu(5,:) = [accup;accups;accuph;accur;accurs;accurh];
