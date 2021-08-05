

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)       
    m = size(X, 1);
    
    num_elem_theta_1 = hidden_layer_size * (input_layer_size + 1);
    # Reshapes theta1 and theta2 into matrices.
    theta1 = reshape(nn_params(1 : num_elem_theta_1), hidden_layer_size, ... 
                     input_layer_size + 1);
    theta2 = reshape(nn_params(num_elem_theta_1 + 1 : end), num_labels, ...
                     hidden_layer_size + 1);
    
    # Feedforwarding performed to obtain the prediction for the calculating the cost.
    [y_hat, a2] = feedforward(theta1, theta2, X);
    
    # Converts y so that each labels is a row vector of 1s and 0s.
    y = eye(num_labels)(y,:);
    
    # Calculates the cost.
    cost_matrix = (y .* log(y_hat)) + ((1 - y) .* log(1 - y_hat));
    reg_term = [theta1(:, 2:end)(:) ; theta2(:, 2:end)(:)];
    reg_term = (lambda / (2 * m)) * sum(reg_term .^ 2); 
    J = (-1/m) * sum(sum(cost_matrix)) + reg_term;
    
    
    # Calculates the gradient
    [grad_theta1, grad_theta2] = backpropagate(theta1, theta2, X, y, lambda);
    grad = [grad_theta1(:); grad_theta2(:)];
    save myfile.mat grad_theta1;
end


# Takes in the weights, the input, and returns the prediction as an m x n matrix.
function [y_hat, a2] = feedforward(theta1, theta2, X)
  m = size(X, 1);
  a1 = [ones(1, m) ; X'];
  a2 = [ones(1, m) ; sigmoid(theta1 * a1)];
  a3 = sigmoid(theta2 * a2);
  y_hat = a3';
  a2 = a2';
end


function [grad_theta1, grad_theta2] = backpropagate(theta1, theta2, X, y, lambda)
  m = size(X, 1);
  n = size(X, 2);
  
  [hx, a2] = feedforward(theta1, theta2, X);
  delta3 = hx - y;            # m x r
  gz = a2(:,2:end) .* (1 - a2(:,2:end));         # (m x h)
  delta2 = theta2(:,2:end)' * delta3' .* gz'; # (h x r) x (r x m) = (h x m)
  
  cap_delta1 = delta2 * [ones(m, 1), X];    # (h x m) x (m x n)   25 x 401
  cap_delta2 = delta3' * a2;    # (m x r) x (m x h) = r x (h + 1)  
  
  grad_theta1 = cap_delta1 / m; 
  grad_theta2 = cap_delta2 / m;
  
  # Regularises the gradients.
  grad_theta1(:,2:end) = grad_theta1(:,2:end) + (lambda / m) * theta1(:,2:end);
  grad_theta2(:,2:end) = grad_theta2(:,2:end) + (lambda / m) * theta2(:,2:end); 
end