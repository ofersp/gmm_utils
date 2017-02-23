% Compute log-likelihood of a GMM at given points.
%
% [log_likelihood,log_resp] = gmm_log_likelihood(X,gmm)
function [log_likelihood,log_resp] = gmm_log_likelihood(X,gmm)

use_gpu = isequal(class(X),'gpuArray');

parfor_arg = Inf; if use_gpu; parfor_arg = 0; end
n = size(X,2);
K = gmm.nmodels;
min_log_resp = -20;

if ~isfield(gmm,'mixweights') % in case we got an uninitialized model  
  log_resp = rand_gpu(n,K,use_gpu)*min_log_resp;
else  
  log_resp = zeros_gpu(n,gmm.nmodels,use_gpu);
  mixweights = gmm.mixweights;
  means = gmm.means;
  covs = gmm.covs;
  parfor (k=1:K, parfor_arg)
    log_resp(:,k) = log(mixweights(k)) + log_gauss_pdf(X,covs(:,:,k),means(:,k)); 
  end
end

log_likelihood = log_sum_exp(log_resp,2);
log_resp = bsxfun(@minus,log_resp,log_likelihood);