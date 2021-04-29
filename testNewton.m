clc;
close all;
clear all;
load('Datasets/realsim.mat');
Xtrain=full(Xtrain);
Xtest=full(Xtest);

tic
[w,b,los,delta]=NewTon(Xtrain,ytrain,100);
toc
plot(los)
title("Newton loss")
z=Xtest*w+b;
sig=1./(1+exp(1).^((-1)*z));
ypred=sign(sig-0.5);
ypred(find(ypred==-1))=0;
accuracy=sum(ypred==ytest)/size(ytest,1);
