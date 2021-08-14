% Computes the validation and training error for different lambda values.
% Takes in X (input), y (labels), lambda_vec (possible lambda values), NUM_ITER 
% (number of iterations).
function [lambda_vec, error_train, error_val] = randomValidationCurve(X, y, lambda_vec, NUM_ITER)
  error_train = zeros(length(lambda_vec), 1);
  error_val = zeros(length(lambda_vec), 1);
  
  HALF_M = floor(size(X, 1) / 2);
  
  % Selects the evaluation and training set separately.
  for i = 1 : NUM_ITER
     new_order = randperm(size(X, 1))';
     X = X(new_order);
     y = y(new_order);
     [i_train_error, i_val_error] = computeErrors(X(1:HALF_M,:), y(1:HALF_M,:), ...
                                                  X(HALF_M+1:end,:), y(HALF_M+1:end,:), lambda_vec);
     error_train = error_train + i_train_error;
     error_val = error_val + i_val_error;
  endfor
  
  error_train = error_train / NUM_ITER;
  error_val = error_val / NUM_ITER;
  
  plotCurves(error_train, error_val, lambda_vec);
end

function [error_train, error_val] = computeErrors(X, y, Xval, yval, lambda_vec)
  % You need to return these variables correctly.
  error_train = zeros(length(lambda_vec), 1);
  error_val = zeros(length(lambda_vec), 1);
  
  for l = 1 : numel(lambda_vec)
    theta = trainLinearReg(X, y, lambda_vec(l));
    error_train(l) = linearRegCostFunction(X, y, theta, 0);
    error_val(l) = linearRegCostFunction(Xval, yval, theta, 0);
  endfor
end 

% Plots the curves on top of each other.
function plotCurves(error_train, error_val, lambda_vec)
  figure
  plot(lambda_vec, error_train);
  hold on;
  plot(lambda_vec, error_val);
  xlabel("Value of lambda");
  ylabel("Error");
  title("Errors for different values of lambda");
  legend("Training", "Validation");
end