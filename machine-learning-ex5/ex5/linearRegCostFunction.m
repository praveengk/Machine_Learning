function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



temp = (X * theta - y).^2

tempTheta = theta;
tempTheta(1) = 0;

temp2 = (tempTheta.^2)*(lambda/ (2 * m));
temp2 = sum(temp2);

J = sum(temp) / (2 * m ) + temp2;


temp1 = (X * theta - y).*X;
temp1 = sum(temp1);
temp1 = temp1';

temp1 = temp1 / m + (lambda / m)*tempTheta;

grad = temp1;


% =========================================================================

grad = grad(:);

end
