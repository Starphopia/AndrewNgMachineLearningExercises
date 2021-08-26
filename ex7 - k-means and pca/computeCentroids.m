function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% Stores the new positions of all of the centroids.
centroids = zeros(K, n);

% Finds the mean of each cluster and repositions the corresponding centroid.
for j = 1 : K
  members = X(idx == j, :);
  centroids(j, :) = sum(members) / size(members, 1);
endfor

end

