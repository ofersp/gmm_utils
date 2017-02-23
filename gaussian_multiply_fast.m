% Product of gaussian densities (faster version).
%
% Compute the unnormalized gaussian resulting from the product -
% N(x;mu_a,cov_a) * N(x;mu_b,cov_b) = c_c * N(x;mu_c,cov_c)
%
% Adapted from Tim Bailey's GMM utilities found here -
% http://www-personal.acfr.usyd.edu.au/tbailey/software/gmm_utilities.htm
%
% [mu_c,cov_c,log_c_c] = gaussian_multiply_fast(mu_a,cov_a,mu_b,cov_b)

function [mu_c,cov_c,log_c_c] = gaussian_multiply_fast(mu_a,cov_a,mu_b,cov_b)

S = cov_a + cov_b;
Sc  = chol(S);

Wc = cov_a/Sc;           % "normalised" gain
vc = ((mu_b-mu_a)'/Sc)'; % "normalised" innovation

% Update 
mu_c = mu_a + Wc*vc; 
cov_c = cov_a - Wc*Wc';

% Update weight
D = size(mu_c,1);
numer = -0.5 * (vc'*vc); 
denom = 0.5*D*log(2*pi) + sum(log(diag(Sc)));
log_c_c = numer - denom;
