function [weights] = stocGradDescent(dataMatrix,classLabels,numsIter)
[m,n] = size(dataMatrix);     % Input matrix is m samples x n dimensions 
weights = ones(n,numsIter); % Default weight matrix n dim x numsIter iterations
for Itr =1:numsIter         % Training Iteration
    for i =1:m              % Sample Iteration
        alpha = 4.0/(i+Itr+1.0)+0.01;   % Dynamic decrease of alpha
        randIndex = randi(m);           % Randomly chose sample
        h = sum(dataMatrix(randIndex,:)*weights);
        h = 1./(1+exp(h));              % h is to predict using w, then take Sigmoid to get lable
        error = classLabels(randIndex)-h;
        weights(:,Itr) = weights(:,Itr) + alpha * error * dataMatrix(randIndex,:)';
                                        % Update weights
    end
end
weights = mean(weights,2);              % Final weight is the mean of all weights
end

