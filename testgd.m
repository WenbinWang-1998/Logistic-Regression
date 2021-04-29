clc;
close all;
clear;
load('Datasets/real-sim.mat');

tic
[wGD,bGD,losGD]=GD(Xtrain,ytrain);
toc

figure()
plot(losGD)
title("GD loss")

zGD=Xtest*wGD+bGD;
sigGD=1./(1+exp(1).^((-1)*zGD));
ypredGD=sign(sigGD-0.5);
ypredGD(find(ypredGD==-1))=0;
accuracyGD=sum(ypredGD==ytest)/size(ytest,1)
