function [X_poly] = polyFeatures(X, p)
for i = 1 : p 
  X_poly(:, i) = X .^ i;
endfor
end
