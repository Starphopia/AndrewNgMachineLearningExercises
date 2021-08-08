function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
  m = length(y); % number of training examples
  
  hx = X * theta;
  jRegTerm = (lambda/(2*m)) * sum([0 ; theta(2:end)] .^ 2);
  J = (1/(2*m)) * sum((hx - y) .^ 2) + jRegTerm;

  gradRegTerm = [0, (lambda/m) * theta(2:end)'];
  grad = (1/m) * sum(X .* (hx - y)) + gradRegTerm;
  grad = grad';
end
