function [w,b,los,delta] = NewTon(X,y,iteration)
[m,d]=size(X);
a=ones(m,1);
X=[a X];
w=zeros(d+1,1);% starting from all zeros
loss=L(X,y,w);%initial lost
iter=1;
delta=inf;
i=1;
while delta>0.01&&iter<=iteration
    jacobian=J(X,y,w);
    hessian=H(X,y,w);
    iter=iter+1;
    w=w-pinv(hessian)*jacobian;
    newloss=L(X,y,w);
    los(i)=newloss;
    i=i+1;
    delta=loss-newloss;
    loss=newloss;
end
b=w(1);
w=w(2:d+1);
end

function [sig] = sigmoid(X,w)
%Calculate the sigmoid function
z=X*w;% m*1
sig=1./(1+exp(1).^((-1)*z));
end

function [loss] = L(X,y,w)%X：m*d
%Lost function
sig=sigmoid(X,w);
loss=(-1)*(sum(y.*log(sig)+(1-y).*log(1-sig)))/size(X,1);
end

function [jacobian] = J(X,y,w)
%calculate Jacobian matrix
sig=sigmoid(X,w);
jacobian=1/size(X,1)*(X'*(sig-y));%d*1
end

function [hessian] = H(X,y,w)
%calculate Hessian matrix
sig=sigmoid(X,w);
%diag(sig)*diag(1-sig);
newsig=sig.*(1-sig);
d=size(X,2);
%hessian=(1/size(X,1))*X'*diag(newsig)*X;%d*d
hessian=(1/size(X,1))*X'.*repmat(newsig',d,1)*X;

end
