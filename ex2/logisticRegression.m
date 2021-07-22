function h = logisticRegression(theta, X)
  h = sigmoid(theta' * X');
end