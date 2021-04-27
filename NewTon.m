function [w,b,los] = NewTon(X,y,iteration)
[m,d]=size(X);
a=ones(m,1);%撇表示转置矩阵
X=[a X];
w=zeros(d+1,1);% starting from all zeros
lost=L(X,y,w);%initial lost
iter=1;
delta=inf;
i=1;
while delta>0.00001&&iter<=iteration 
    tic
    jacobian=J(X,y,w);
    hessian=H(X,y,w);
    iter=iter+1;
    w=w-pinv(hessian)*jacobian;
    newlost=L(X,y,w)
    los(i)=newlost;
    i=i+1;
    delta=lost-newlost;
    lost=newlost;   
    toc
end
b=w(1);
w=w(2:d+1);
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
hessian=(1/size(X,1))*X'*diag(sig)*diag(1-sig)*X;%d*d
end

% function [hessian] = H(X,y,w)
% %calculate Hessian matrix
% sig=sigmoid(X,w);
% hessian=(1/size(X,1))*(sig'*(1-sig))*(X'*X);%d*d
% end
