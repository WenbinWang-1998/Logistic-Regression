clc;
close all;
clear all;
load('adult_train_test.mat');
Xtrain=full(Xtrain);
Xtest=full(Xtest);
ytrain(find(ytrain==-1))=0;
ytest(find(ytest==-1))=0;
y=ytest;

[w]=Conjugate_Gradient(Xtrain,ytrain);
b=w(1);
w=w(2:size(Xtrain,2)+1);
z=Xtest*w+b;
sig=1./(1+exp(1).^((-1)*z));
ypred=sign(sig-0.5);
accuracy=sum(ypred==y)/size(y,1)