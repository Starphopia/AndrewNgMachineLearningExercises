function [J, grad] = costFunctionReg(theta, X, y, lambda)
  %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization

  % Initialize some useful values
  m = length(y); % number of training examples
  original_theta = theta;
  original_theta(1) = 0;

  [J, grad] = costFunction(theta, X, y);

  % Regularizes the cost and gradient.
  grad = grad + (lambda / m) * original_theta;
  J = J + (lambda / (2 * m)) * sum(original_theta .^ 2);
end
