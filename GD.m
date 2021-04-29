function [weights,bias,loss] = GD(Xtr,Ytr)
[m,n] = size(Xtr);     % Input matrix is m samples x n dimensions 
Xtil=[ones(m,1) Xtr];
W=zeros(n,1);
Wtil=[0; W];
Wsum=[Wtil];
delta = 0.0023;
lossdiff = 2*delta;
Itr=0;
while lossdiff > delta             % Training Iteration
    Itr=Itr+1;
    alpha = 4.0/(Itr+1.0)+0.01;
    sum = zeros(n+1,1);
    for i =1:m              % Sample Iteration
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
end
weights = Wtil(2:n+1);
bias = Wtil(1);
end

function [sig] = sigmoid(X,w)
%Calculate the sigmoid function
z=X*w;
sig=1./(1+exp(1).^((-1)*z));
end

function [loss] = L(X,y,w)
%Loss function
sig=sigmoid(X,w);
loss=(-1)*(sum(y.*log(sig)+(1-y).*log(1-sig)))/size(X,1);
end

