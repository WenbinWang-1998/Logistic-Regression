function [w,b,loss] = Conjugate_Gradient(X,y)
[m,length] = size(X);
a = ones(m,1);
X = [a X];
w = zeros(length+1,1);

c1 = 0.01;                                            % c1 for Armijo condition
c2 = 0.5;                                          % c2 for curvature condition
i = 0;
loss = [];
[f1,df1] = func_diff(X,y,w);
s = -df1;                                           % steepest search direction
d1 = -s'*s;
alpha = 1/(1-d1);                                 % initial step size 1/(|s|+1)

while i < length
    w = w + alpha*s;
    [f2,df2] = func_diff(X,y,w);
    d2 = df2'*s;
    while (f2 > f1+alpha*c1*d1) || (d2 > -c2*d1)             % Wolfe conditions
        alpha = 0.5*alpha;
        if alpha < 10*eps
            error('Line search failed - alpha close to working precision');
        end
        w = w + alpha*s;
        [f2,df2] = func_diff(X,y,w);
        d2 = df2'*s;
    end
    if f1-f2 < 0.0001
        break
    end
    f1 = f2; loss = [loss' f1]';
    s = (df2'*df2)/(df1'*df1)*s - df2;              % Fletcher-Reeves direction
%     s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;      % Polack-Ribiere direction
    df1 = df2;
    d2 = df1'*s;
    if d2 > 0
      s = -df1;
      d2 = -s'*s;    
    end
    d1 = d2;
    i = i + 1;
end
b = w(1);
w = w(2:length+1);
end

function [f,df] = func_diff(X,y,w)
z = X*w;
sig = 1./(1+exp(1).^((-1)*z));
f = (-1)*(sum(y.*log(sig)+(1-y).*log(1-sig)))/size(X,1);
df = 1/size(X,1)*(X'*(sig-y));
end