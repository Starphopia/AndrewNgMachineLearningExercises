% Plotting the profit to population data
data = load('ex1data1.txt'); % read comma separated data
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples
plotData(X, y)

% Computing the cost.
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
iterations = 1500;
alpha = 0.01;

% Compute and display initial cost with zero theta
computeCost(X, y, theta)

% Compute and display initial cost with non-zero theta
computeCost(X, y,[-1; 2])

% Run gradient descent:
% Compute theta
[theta, history] = gradientDescent(X, y, theta, alpha, iterations);



% Print theta to screen
% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f,\n%f',theta(1),theta(2))

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n', predict1*10000);

predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n', predict2*10000);

figure(2)
plot(history, 1:size(history))