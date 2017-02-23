% Expand a low dimensional GMM to model a higher dimensional sample.
%
% gmm_out = gmm_expand(gmm_in,batch,in_dim_subset,reg_var)
% gmm_out = gmm_expand(gmm_in,batch,in_dim_subset,reg_var,reg_type)
function gmm_out = gmm_expand(gmm_in,batch,in_dim_subset,reg_var,reg_type)

if nargin() < 5; reg_type = 'covs_add'; end

if isstr(batch)
  fprintf(1,'Reading training samples...\n');
  batch = batch_load(batch);
end
X = batch.samples;
clear batch;

d = size(X,1);
n = size(X,2);
K = gmm_in.nmodels;
gmm_out.nmodels = K;
gmm_out.dim = d;
means = zeros(d,K);
covs = zeros(d,d,K);

% E step
fprintf(1,'Performing E-step using gmm_in...\n');
[~,log_resp] = gmm_log_likelihood(X(in_dim_subset,:),gmm_in);

% M step
fprintf(1,'Performing M-step to produce gmm_out...\n');
log_sum_resp = log_sum_exp(log_resp);
mixweights = exp(log_sum_resp)'/n;
parfor k=1:K   
  W = spdiags(exp(log_resp(:,k)-log_sum_resp(k)),0,n,n);
  means(:,k) = sum(X*W,2);
  cov_curr = X*W*X' - means(:,k)*means(:,k)';
  covs(:,:,k) = regularized_cov(cov_curr,reg_var,reg_type);
end

% update gmm fields
gmm_out.means = means;
gmm_out.covs = covs;
gmm_out.mixweights = mixweights;

% keep some metadata
gmm_out.metadata.reg_var = reg_var;
gmm_out.metadata.reg_type = reg_type;
gmm_out.metadata.used_gpu = false;

% possibly fix the auxilary fields
gmm_out = gmm_fix(gmm_out);
