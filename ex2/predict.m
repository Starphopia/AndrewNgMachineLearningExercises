function p = predict(theta, X)
  %PREDICT Predict whether the label is 0 or 1 using learned logistic 
  %regression parameters theta

  m = size(X, 1); % Number of training examples
  p = zeros(m, 1)
  
  % Records all positive examples.
  p(find(logisticRegression(theta, X) >= 0.5)) = 1
end
