clc;
close all;
clear all;
load('adult_train_test.mat');
Xtrain=full(Xtrain);
Xtest=full(Xtest);
y=ytest;
ytest(find(ytest==-1))=0;

[w,b]=NewTon(Xtrain,ytrain,1);
z=Xtest*w+b;
sig=1./(1+exp(1).^((-1)*z));
ypred=sign(sig-0.5);
accuracy=sum(ypred==y)/size(y,1);
