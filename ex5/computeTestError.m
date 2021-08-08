function [testError] = computeTestError(X, y, Xtest, ytest, Xval, yval)
[lambda_vec, error_train, error_val] = validationCurve(X, y, Xval, yval);
[~, best] = min(error_val);
theta = trainLinearReg(X, y, lambda_vec(best));
testError = linearRegCostFunction(Xtest, ytest, theta, 0)
end