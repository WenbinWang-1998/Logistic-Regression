%% First test
clc;
close all;
clear all;
load('w1a.mat');
Xtrain=full(Xtrain);
Xtest=full(Xtest);
%y=ytest;
%ytest(find(ytest==-1))=0;

[w,b,los]=NewTon(Xtrain,ytrain,5);
plot(los)
z=Xtest*w+b;
sig=1./(1+exp(1).^((-1)*z));
ypred=sign(sig-0.5);
ypred(find(ypred==-1))=0;
accuracy=sum(ypred==ytest)/size(ytest,1);

%% Second test(plot two dimensional dataset)
Xtrain=ex4x;
ytrain=ex4y;
[m,d]=size(Xtrain);
% w=NewTon(Xtrain,ytrain,1);
% Xtrain=[ones(m,1) Xtrain];
pos = find ( ytrain == 1 ) ; neg = find ( ytrain == 0 ) ;
% Assume t h e f e a t u r e s are in t h e 2nd and 3 rd
% columns o f x
plot ( Xtrain( pos , 1) , Xtrain( pos , 2 ) , '+' ) ; hold on
plot ( Xtrain( neg , 1 ) , Xtrain( neg , 2 ) , 'o' ); hold on
[w,b,los]=NewTon(Xtrain,ytrain,10);
% p=round(1./(1+exp(1).^((-1)*(Xtrain*w+b))));
% accuracy=mean(double(p==ytrain)*100)
max_value = max(Xtrain(:,1));
min_value = min(Xtrain(:,1));
X = min_value:0.001:max_value;
Y = -(b + w(1,1) * X) / w(2,1);
plot(X, Y, '-')


%% Third test titanic
clc;
close all;
clear all;
load('titanic.mat');
[w,b,los]=NewTon(Xtr,ytr,5);
w
plot(los)
z=Xtr*w+b;
sig=1./(1+exp(1).^((-1)*z));
ypred=sign(sig-0.5);
ypred(find(ypred==-1))=0;
accuracy=sum(ypred==ytr)/size(ytr,1)
