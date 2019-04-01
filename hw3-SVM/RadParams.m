function [C, Ccandi, perror] = RadParams(X, y, sigma)

Ccandi = [0.01,0.03,0.1,0.3,1,3,10,30,100];

perror = zeros(length(Ccandi),1);
l = size(X,1);
Xval = X(1:2:2*floor(l/2),:);
yval = y(1:2:2*floor(l/2));
X = X(2:2:2*floor(l/2)-1,:);
y = y(2:2:2*floor(l/2)-1);

for i = 1:length(Ccandi)
        model= RadTrain(X, y, Ccandi(i), sigma,[],5);
        predictions = RadPredict(model, Xval);
        perror(i) = mean(double(predictions ~= yval));
end

[val,idx] = min(perror(:));
I = ind2sub(size(perror),idx);
C = Ccandi(I);

end