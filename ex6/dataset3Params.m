function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% function [model] = svmTrain(X, Y, C, kernelFunction, ...
%                            tol, max_passes)


lowest_error = intmax;

val_arr = [ 0.01 0.03 0.1 0.3 1 3 10 30 ];

for i = 1: length(val_arr)
    for j = 1:length(val_arr)
        model = svmTrain(X,y,val_arr(i), @(x1,x2) gaussianKernel(x1,x2,val_arr(j)));

        predictions = svmPredict(model,Xval);

        pred_error = mean(double(predictions ~= yval));
        
        if pred_error < lowest_error
            C = val_arr(i);
            sigma = val_arr(j);
            lowest_error = pred_error;
        end
    end
end







% =========================================================================

end
