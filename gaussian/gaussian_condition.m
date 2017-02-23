% Condition a gaussian variable on a linear equality constraint.
%
% Given distribution x ~ N(mu,cov) and matrix A defining the constraint 
% y = Ax, compute the parameters of the gaussian distribution x|Ax=y
%
% [mu_c,cov_c] = gaussian_condition(mu,cov,A,y)

function [mu_c,cov_c] = gaussian_condition(mu,cov,A,y)

AcovAt = A*cov*A';
covAt = cov*A';
mu_c = mu + covAt*(AcovAt\(y-A*mu));
cov_c = cov - covAt*(AcovAt\covAt');