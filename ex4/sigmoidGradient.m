function g = sigmoidGradient(z)
  gz = sigmoid(z);
  g = gz .* (1 - gz);
end
