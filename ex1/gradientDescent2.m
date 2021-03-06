function [theta, J_history] = gradientDescent2(X,y,theta,alpha,num_iters)

m = length(y);
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
    
    
    



