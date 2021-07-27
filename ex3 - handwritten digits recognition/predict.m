function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);


% Appends the column of 1s to X.
X = [ones(m, 1), X]; % m x n + 1

a2 = sigmoid(Theta1 * X'); % 25 x m   
a2 = [ones(1, m); a2]; % 26 x m
a3 = sigmoid(Theta2 * a2); % numbersoflabels x 26
[~, p] = max(a3', [], 2);
size(p)
p






end
