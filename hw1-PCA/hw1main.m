% 02/05/2019
% yue wu (sophy)
% hw1
% CS 391L

clear all;close all;

% store training set into column vectors
load digits.mat
trainset = reshape(im2double(trainImages),[],60000);
testset = reshape(im2double(testImages),[],10000);

% find eigendigits and experiments
% ntrain = [1000,2000,5000,10000,20000];
ntrain = 2000;
A = trainset(:,1:ntrain);
[m,V] = hw1FindEigendigits(A);
projtest = real((testset-m)'*V);
projtrain = real((trainset(:,1:ntrain)-m)'*V);
% neig = 30;
neig = 50:50:750;
% recons = reshape((V(:,1:neig)*projtest(2,1:neig)'+m),28,28);
% imshow(recons)
for i = 1:length(neig)
    i
    projtraincut = projtrain(:,1:neig(i));
    classtraincut = trainLabels(1:ntrain);
    Mdl = fitcknn(projtraincut,classtraincut,'NumNeighbors',5);
    projtestcut = projtest(:,1:neig(i));
    label = predict(Mdl,projtestcut);
    accuracy(i) = sum(label' == testLabels)/10000;
end

figure
plot(neig,accuracy,'linewidth',2)
title(['# of Training Sample = 30000'])
xlabel("# of eigenvalues")
ylabel("accuracy [%]")
ylim([0.6 1])
grid on
set(findall(gcf,'-property','FontSize'),'FontSize',16)
