%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.
function [J, grad] = costFunction(theta, X, y)
  % Initialize some useful values
  m = length(y); % number of training examples

  % You need to return the following variables correctly
  hx = logisticRegression(theta, X);
  JMatrix = y' .* log(hx) + (1 - y)' .* log(1 - hx); 
  J = (-1/m) * sum(JMatrix(:));
  grad = (1/m)*X'*(hx - y')';
end

