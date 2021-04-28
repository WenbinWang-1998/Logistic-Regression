function [weights,bias,loss,t,lossdiff] = GD(Xtr,Ytr,numsIter)
[m,n] = size(Xtr);     % Input matrix is m samples x n dimensions 
% weights = zeros(n,1); % Default weight matrix n dim x numsIter iterations

Xtil=[ones(m,1) Xtr];
W=zeros(n,1);
Wtil=[0; W];
loss = zeros(1,numsIter);
Wsum=[Wtil];
t=zeros(1,numsIter);
delta = 0.001;
lossdiff = 2*delta;
Itr=0;
while Itr < numsIter && lossdiff > delta             % Training Iteration
    tic
    Itr=Itr+1;
    alpha = 4.0/(Itr+1.0)+0.01;
    sum = zeros(n+1,1);
    for i =1:m              % Sample Iteration (not needed because SGD
        h = sigmoid(Xtil(i,:),Wtil);
        error =  h - Ytr(i);
        sum = sum +  error * Xtil(i,:)';
    end
    Wtil = Wtil - alpha *sum/m;
    Wsum = Wsum+Wtil;
    loss(Itr) = L(Xtil,Ytr,Wsum/(Itr));
    if Itr>1
        lossdiff = loss(Itr-1)-loss(Itr);
    end
    t(Itr)=toc;
end
% Wtil = Wsum/numsIter/m;
% weights = mean(weights,2);              % Final weight is the mean of all weights
weights = Wtil(2:n+1);
bias = Wtil(1);
loss=loss(1:Itr);
t=t(1:Itr);
end

function [sig] = sigmoid(X,w)
%Calculate the sigmoid function
z=X*w;% m*1
sig=1./(1+exp(1).^((-1)*z));
end

function [lost] = L(X,y,w)%X?m*d
%Lost function
sig=sigmoid(X,w);
lost=(-1)*(sum(y.*log(sig)+(1-y).*log(1-sig)))/size(X,1);%???m
end

