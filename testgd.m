% test 1
clc;
close all;
clear all;
load('./Data/a9a.mat');
Xtrain=full(Xtrain);
Xtest=full(Xtest);
y=ytest;
ytest(find(ytest==-1))=0;
itersSGD=100;
itersGD =10;

[wGD,bGD,losGD,tGD,deltaGD]=GD(Xtrain,ytrain,itersGD);
[wSGD,bSGD,losSGD,tSGD,deltaSGD]=stocGradDescent(Xtrain,ytrain,itersSGD);
% losGD
% losSGD
GDtime=sum(tGD)
SGDtime=sum(tSGD)


figure()
plot(losGD)
title("GD loss")
% figure()
% plot(tGD)
% title("GD Iteration Time")

zGD=Xtest*wGD+bGD;
zSGD=Xtest*wSGD+bSGD;
sigGD=1./(1+exp(1).^((-1)*zGD));
sigSGD=1./(1+exp(1).^((-1)*zSGD));

ypredGD=sign(sigGD-0.5);
ypredSGD=sign(sigSGD-0.5);

ypredGD(find(ypredGD==-1))=0;
ypredSGD(find(ypredSGD==-1))=0;

accuracyGD=sum(ypredGD==y)/size(y,1)
accuracySGD=sum(ypredSGD==y)/size(y,1)
deltaGD
deltaSGD

