% Product of gaussian densities.
%
% Compute the unnormalized gaussian resulting from the product -
% N(x;mu_a,cov_a) * N(x;mu_b,cov_b) = c_c * N(x;mu_c,cov_c)
%
% See Matrix-Cookbook sec. 8.1.8 eq. 371.
%
% [mu_c,cov_c,log_c_c] = gaussian_multiply(mu_a,cov_a,mu_b,cov_b)

function [mu_c,cov_c,log_c_c] = gaussian_multiply(mu_a,cov_a,mu_b,cov_b)

log_c_c = log_gauss_pdf(mu_a,cov_a+cov_b,mu_b);
inv_cov_a = inv(cov_a);
inv_cov_b = inv(cov_b);
inv_cov_c = inv_cov_a+inv_cov_b;
cov_c = inv(inv_cov_c); 
mu_c = inv_cov_c\(cov_a\mu_a + cov_b\mu_b);
