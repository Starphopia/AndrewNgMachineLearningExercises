function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% The clusters each example will be assigned to.
idx = zeros(size(X,1), 1);

% Set K
K = size(centroids, 1);
m = size(X, 1);

for i = 1 : m
  # Norm computes the distance of a vector point from the origin.
  distances = norm(centroids - X(i,:), 2, "rows");
  [~, idx(i)] = min(distances);
endfor
end
