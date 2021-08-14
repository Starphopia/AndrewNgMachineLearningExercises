function sim = gaussianKernel(x1, x2, sigma)
  % Ensures x1 and x2 are column vectors 
  x1 = x1(:);
  x2 = x2(:);

  distance_squared = (x1 - x2) .^ 2;
  sim = exp(-sum(distance_squared) / (2 * sigma ^ 2));
end
