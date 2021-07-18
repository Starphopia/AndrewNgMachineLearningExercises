function J = computeCost(X, y, weights)
  m = length(y); % Number of training examples
  sqrDiffs = (predict(X, weights) - y) .^ 2;
  J = (0.5 / m) * sum(sqrDiffs(:));

end

function yHat = predict(X, weights) 
  yHat = X * weights;
end 
