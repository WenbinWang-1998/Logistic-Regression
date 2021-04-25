function [weights,bias,loss] = stocGradDescent(Xtr,Ytr,numsIter)
[m,n] = size(Xtr);     % Input matrix is m samples x n dimensions 
% weights = zeros(n,1); % Default weight matrix n dim x numsIter iterations

Xtil=[ones(m,1) Xtr];
W=zeros(n,1);
Wtil=[1; W];
loss = zeros(1,numsIter);
for Itr =1:numsIter         % Training Iteration
    for i =1:m              % Sample Iteration (not needed because SGD
        % Dynamic decrease learning rate alpha
        alpha = 4.0/(i+Itr+1.0)+0.01;   
%         alpha = 0.01; % fixed learning rate

        randIndex = randi(m);           % Randomly chose sample
        Xt=Xtil(randIndex,:);
        Yt=Ytr(randIndex);
        
        %  predict using w, take Sigmoid to get lable
        h = sigmoid(Xt,Wtil);

        error =  h - Yt;
        Wtil = Wtil - alpha * error * Xt';
                                        % Update weights
    end

    loss(Itr) = L(Xtil,Ytr,Wtil);
end
% weights = mean(weights,2);              % Final weight is the mean of all weights
weights = Wtil(2:n+1);
bias = Wtil(1);
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

