function [C, Ccandi, perror] = PolyParams(X, y, p, l)

Ccandi = [0.01,0.03,0.1,0.3,1,3,10,30,100];

perror = zeros(length(Ccandi),1);
L = size(X,1);
Xval = X(1:2:2*floor(L/2),:);
yval = y(1:2:2*floor(L/2));
X = X(2:2:2*floor(L/2)-1,:);
y = y(2:2:2*floor(L/2)-1);

for i = 1:length(Ccandi)
        model= PolyTrain(X, y, Ccandi(i), p, l, [], 5);
        predictions = PolyPredict(model, Xval);
        perror(i) = mean(double(predictions ~= yval));
end

[val,idx] = min(perror(:));
I = ind2sub(size(perror),idx);
C = Ccandi(I);

end