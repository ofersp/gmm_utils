% Train a gaussian-mixture-model (GMM) using expectation-maximization.
%
% [gmm,log_likelihood] = gmm_ml_training(X,gmm0,num_iters,reg_var)
% [gmm,log_likelihood] = gmm_ml_training(X,gmm0,num_iters,reg_var,use_gpu)
% [gmm,log_likelihood] = gmm_ml_training(X,gmm0,num_iters,reg_var,use_gpu,reg_type)
% [gmm,log_likelihood] = gmm_ml_training(X,gmm0,num_iters,reg_var,use_gpu,reg_type,zero_means)
%
% See: regularized_cov.m for possible reg_type values.
function [gmm,log_likelihood] = gmm_ml_training(X,gmm0,num_iters,reg_var,use_gpu,reg_type,zero_means)

if nargin() < 7; zero_means = false; end
if nargin() < 6; reg_type = 'covs_add'; end
if nargin() < 5; use_gpu = false; end

parfor_arg = Inf; if use_gpu; parfor_arg = 0; end
d = size(X,1);
n = size(X,2);
K = gmm0.nmodels;
gmm = gmm0; clear gmm0;
means = zeros_gpu(d,K,use_gpu);
covs = zeros_gpu(d,d,K,use_gpu);
if use_gpu; X = gpuArray(X); end

for j=1:num_iters
  % E step
  [log_likelihood,log_resp] = gmm_log_likelihood(X,gmm);  
  % M step
  log_sum_resp = log_sum_exp(log_resp);
  mixweights = exp(log_sum_resp)'/n;
  %parfor (k=1:K, parfor_arg)
  for k=1:K
    W = spdiags(exp(log_resp(:,k)-log_sum_resp(k)),0,n,n);
    if zero_means
      cov_curr = X*W*X';
    else
      means(:,k) = sum(X*W,2);
      cov_curr = X*W*X' - means(:,k)*means(:,k)';
    end
    covs(:,:,k) = regularized_cov(cov_curr,reg_var,reg_type);
  end
  % update gmm fields
  gmm.means = means;
  gmm.covs = covs;
  gmm.mixweights = mixweights;
end

% keep some metadata
gmm.metadata.reg_var = reg_var;
gmm.metadata.reg_type = reg_type;
gmm.metadata.n_em_iters = num_iters;
gmm.metadata.used_gpu = use_gpu;
gmm.metadata.zero_means = zero_means;