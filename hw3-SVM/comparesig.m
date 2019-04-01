function [accu] = comparesig(A,ALabels,Cr,sigcandi,B,BLabels)

accu = zeros(length(sigcandi),1);
for i = 1:length(sigcandi)
    rmodel = RadTrain(A, ALabels, Cr, sigcandi(i), [], 20);
    pred = RadPredict(rmodel,B);
    accu(i) = sum(pred==BLabels)/length(BLabels);
end

end