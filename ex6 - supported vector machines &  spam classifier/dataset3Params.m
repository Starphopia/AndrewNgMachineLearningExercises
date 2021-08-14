function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel

% Values to try for sigma and C.
toTry = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
numToTry = length(toTry);


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
costs = zeros(numToTry,numToTry);


% Loops through to compute the cost for different sigma and C values.
for c = 1 : numToTry
  for s = 1 : numToTry
    model = svmTrain(X, y, toTry(c), @(x1, x2) gaussianKernel(x1, x2, toTry(s)));  
    predictions = svmPredict(model, Xval);
    costs(c, s) = mean(double(predictions ~= yval));    
  endfor
endfor

% Gets the optimal C and sigma values.
[CIndex, sigmaIndex] = find(costs == min(min(costs)));
C = toTry(CIndex);
sigma = toTry(sigmaIndex)
end
