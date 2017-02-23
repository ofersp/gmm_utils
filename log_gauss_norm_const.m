% Evaluate the logarithmic normaliztion constant of given gaussian covariance
%
% log_norm_const = log_gauss_norm_const(sigma)
function log_norm_const = log_gauss_norm_const(sigma)

d = size(sigma,1);

[U,p]= chol(sigma);
if p ~= 0; error('sigma is not positive-definite.'); end

c = d*log(2*pi)+2*sum(log(diag(U))); % normalization constant
log_norm_const = -c/2;

