% %test 1
% clc;
% close all;
% clear all;
% load('./Data/w1a.mat');
% Xtrain=full(Xtrain);
% Xtest=full(Xtest);
% y=ytest;
% ytest(find(ytest==-1))=0;
% tic
% [w,b,los]=stocGradDescent(Xtrain,ytrain,100);
% toc
% los
% hold
% plot(los)
% z=Xtest*w+b;
% sig=1./(1+exp(1).^((-1)*z));
% if sig>0.5
%     ypred=1;
% else
%     ypred=0;
% end
% % ypred=sign(sig-0.5);
% accuracy=sum(ypred==y)/size(y,1)


%test 2
clc;
% close all;
clear all;
load('./Data/titanic.mat');
[w,b,los]=stocGradDescent(Xtr,ytr,100);
w
hold
plot(los)
z=Xte*w+b;
sig=1./(1+exp(1).^((-1)*z));
ypred=sign(sig-0.5);
% if sig>0.5
%     ypred=1;
% else
%     ypred=0;
% end
ypred(find(ypred==-1))=0;
accuracy=sum(ypred==yte)/size(yte,1)
