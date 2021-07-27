% http://www.massmind.org/techref/method/ai/fmincg.htm
% http://www.ece.northwestern.edu/local-apps/matlabhelp/techdoc/ref/optimset.html

function [all_theta] = oneVsAll(X, y, num_labels, lambda)
% Some useful variables
m = size(X, 1); % number of rows
n = size(X, 2); % number of columns

% You need to return the following variables correctly 
all_theta = [];

% Add ones to the X data matrix
X = [ones(m, 1) X];

% An initial guess for theta.
initial_theta = zeros(n + 1, 1);

% Options for fmincg
options = optimset('GradObj', 'on', 'Display', 'notify', 'MaxIter', 50);

for k = 1 : num_labels
  [theta_k] = fmincg(@(t) lrCostFunction(t, X, (y==k), lambda), initial_theta, ...
                     options);
  all_theta = [all_theta ; theta_k'];
end 

end
