function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

# Unroll parameter vector to obtain theta and X.
X = reshape(params(1:num_features * num_movies), num_movies, num_features);
Theta = reshape(params(num_features * num_movies + 1:end), num_users, num_features);

# Compute the cost.
J = 0.5 * sum(sum(((X * Theta' - Y) .* R) .^ 2)) + 0.5 * lambda * sum((Theta .^ 2)(:)) ...
    + 0.5 * lambda * sum((X .^ 2)(:));

# Computes the gradients.
X_grad = ((X * Theta' - Y) .* R) * Theta + lambda * X;
Theta_grad = ((X * Theta' - Y) .* R)' * X + lambda * Theta; 
grad = [X_grad(:); Theta_grad(:)];

end
