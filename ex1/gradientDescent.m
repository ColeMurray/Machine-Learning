function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters+1
    hypothesis = X * theta; % m x 1 product
    errorsV = hypothesis - y ;
    
    theta_change = (X'*errorsV) * alpha * (1/m); % n x 1
    
    theta = theta - theta_change;
    
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end




end
