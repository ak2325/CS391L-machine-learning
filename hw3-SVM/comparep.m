function [accu] = comparep(A,ALabels,Cp,lcandi,pcandi,B,BLabels)

accu = zeros(length(pcandi),length(lcandi));
for i = 1:length(pcandi)
    for j = 1:length(lcandi)
        pmodel = PolyTrain(A, ALabels, Cp, pcandi(i), lcandi(j), [], 20);
        pred = PolyPredict(pmodel,B);
        accu(i,j) = sum(pred==BLabels)/length(BLabels);
    end
end

end