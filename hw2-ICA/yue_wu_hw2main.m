% 02/19/2019
% yue wu (sophy)
% hw2
% CS 391L

clear all;close all;

% get data
load sounds.mat
T = 1:44000;
U = sounds(1:5,T);
[n,t] = size(U);
m = 8;
A = rand(m,n);
X = A*U;

% load icaTest.mat
% [n,t] = size(U);
% m=3;
% X = A*U;

% implement ICA algorithm
W = rand(n,m)./10;
eta = 0.1;
ite = 100000;
for i = 1:ite
    if mod(i,1000) == 0
        i
    end
    Y = W*X;
    Z = 1./(1+exp(-Y));
    dW = eta*(eye(n)+(1-2*Z)*Y'./t)*W;
    W = W+dW;
%     sqdiff = (U-Y).^2;
%     diff(i) = sum(sqdiff(:));
    cov = abs(dW*X);
    covjudge(i) = max(cov(:));
    if covjudge(i)<10e-10
        Nite = i;
        break;
    end
end
Yview(1:n,:) = 2*(Y(1:n,:)-min(Y,[],2))./(max(Y,[],2)-min(Y,[],2))-1;
% Utemp(1:n,:) = 2*(U(1:n,:)-min(U,[],2))./(max(U,[],2)-min(U,[],2))-1;

% correct for sequence and normalize recovered signal
% another way to do this: cross multiplication (won't be affected by change in sign)
% compare abs(U'*T)/(norm(U)*norm(Y))

Ynorm = zeros(n,t);
for k = 1:n
    Utemp = U(k,:);
    Ytempnormed = zeros(n,t);
    corr = zeros(n,1);
    for j = 1:n
        Ytempnormed(j,:) = Y(j,:)*norm(Utemp)/norm(Y(j,:));
        corrtemp = corrcoef(Utemp,Ytempnormed(j,:));
        corr(j) = abs(corrtemp(1,2));
    end
    [val,idx] = max(corr);
    idx
    Ynorm(k,:) = Ytempnormed(idx,:);
end

figure(1)
plot(1:i,covjudge,'linewidth',2)
title(sprintf('Sound Set, eta = %f, i = %d',eta,Nite))
% xlabel("iterations")
% ylabel("sum of square of differences")
grid on
set(findall(gcf,'-property','FontSize'),'FontSize',16)

figure(2)
plot(T,U(1,:)+1)
hold on
plot(T,U(2,:)+3)
hold on
plot(T,U(3,:)+5)
hold on
plot(T,U(4,:)+7)
hold on
plot(T,U(5,:)+9)
ylim([0 10])
xlim([0 T(end)])
ylabel('original signal')
xlabel('time')
grid on
set(findall(gcf,'-property','FontSize'),'FontSize',16)
hold off

figure(3)
plot(T,Ynorm(1,:)+1)
hold on
plot(T,Ynorm(2,:)+3)
hold on
plot(T,Ynorm(3,:)+5)
hold on
plot(T,Ynorm(4,:)+7)
hold on
plot(T,Ynorm(5,:)+9)
hold off
ylim([0 10])
xlim([0 T(end)])
ylabel('recovered and normalized signal')
xlabel('time')
grid on
set(findall(gcf,'-property','FontSize'),'FontSize',16)

figure(4)
plot(T,Yview(1,:)+1)
hold on
plot(T,Yview(2,:)+3)
hold on
plot(T,Yview(3,:)+5)
hold on
plot(T,Yview(4,:)+7)
hold on
plot(T,Yview(5,:)+9)
hold off
ylim([0 10])
xlim([0 T(end)])
ylabel('recovered signal without correcting for sequence')
xlabel('time')
grid on
set(findall(gcf,'-property','FontSize'),'FontSize',16)
