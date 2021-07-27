function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); % number of training examples

% You need to return the following variables correctly with theta being a column
% vector.
hx = sigmoid(X * theta);

regPart = (lambda / (2*m)) * sum([0 ; theta(2:end)] .^ 2);
costs = -y .* log(hx) - (1 - y) .* log(1 - hx);
J = (1 / m) * sum(costs(:)) + regPart;

regPart = (lambda / m) * [0 ; theta(2:end)];
grad = (1 / m) * (X' * (hx - y)) + regPart;
end
