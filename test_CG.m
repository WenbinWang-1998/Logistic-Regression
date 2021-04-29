clc;
clear;
close all;
load('Datasets/real-sim.mat');

tic
[w,b,loss] = Conjugate_Gradient(Xtrain,ytrain);
toc
plot(loss)
title("CG loss")
z = Xtest*w+b;
sig = 1./(1+exp(1).^((-1)*z));
ypred = sign(sig-0.5);
ypred(ypred==-1) = 0;
Accuracy = sum(ypred==ytest)/size(ytest,1)