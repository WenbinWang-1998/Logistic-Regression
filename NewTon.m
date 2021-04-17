function [w] = NewTon(X,y,iteration)
[m,d]=size(X);
w=zeros(d,1);% starting from all zeros
lost=L(X,y,w);%initial lost
iter=1;
delta=inf;
while delta>0.00001&&iter<=iteration 
    jacobian=J(X,y,w);
    hessian=H(X,y,w);
    iter=iter+1;
    w=w-pinv(hessian)*jacobian;
    newlost=L(X,y,w)
    delta=lost-newlost;
    lost=newlost;
end
end

function [sig] = sigmoid(X,w)
%Calculate the sigmoid function
z=X*w;% m*1
sig=1./(1+exp(1).^((-1)*z));
end

function [lost] = L(X,y,w)%X：m*d
%Lost function
sig=sigmoid(X,w);
lost=(-1)*(sum(y.*log(sig)+(1-y).*log(1-sig)))/size(X,1);%别忘除m
end

function [jacobian] = J(X,y,w)
%calculate Jacobian matrix
sig=sigmoid(X,w);
jacobian=1/size(X,1)*(X'*(sig-y));%d*1
end

function [hessian] = H(X,y,w)
%calculate Hessian matrix
sig=sigmoid(X,w);
hessian=(1/size(X,1))*(sig'*(1-sig))*(X'*X);%d*d
end
