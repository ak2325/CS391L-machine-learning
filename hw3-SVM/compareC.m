function [accup, accuph, accups, accur, accurs, accurh] = compareC(A, ...
    ALabels, Cp, p, l, Cr, sigma, B, BLabels)

% Polynomial SVM
pmodel = PolyTrain(A, ALabels, Cp, p, l, [], 20);
Cps = 0.01;
Cph = 10000;
pmodelsoft = PolyTrain(A, ALabels, Cps, p, l, [], 20);
pmodelhard = PolyTrain(A, ALabels, Cph, p, l, [], 20);

pred = PolyPredict(pmodel,B);
accup = sum(pred==BLabels)/length(BLabels);
preds = PolyPredict(pmodelsoft,B);
accups = sum(preds==BLabels)/length(BLabels);
predh = PolyPredict(pmodelhard,B);
accuph = sum(predh==BLabels)/length(BLabels);

% Radial Basis SVM
rmodel = RadTrain(A, ALabels, Cr, sigma, [], 20);
Crs = 0.01;
Crh = 10000;
rmodelsoft = RadTrain(A, ALabels, Crs, sigma, [], 20);
rmodelhard = RadTrain(A, ALabels, Crh, sigma, [], 20);

pred = RadPredict(rmodel,B);
accur = sum(pred==BLabels)/length(BLabels);
preds = RadPredict(rmodelsoft,B);
accurs = sum(preds==BLabels)/length(BLabels);
predh = RadPredict(rmodelhard,B);
accurh = sum(predh==BLabels)/length(BLabels);

end