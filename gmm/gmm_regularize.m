% Regularize all of GMM components.
%
% gmm_r = gmm_regularize(gmm,reg_var,reg_type)

function gmm_r = gmm_regularize(gmm,reg_var,reg_type)

gmm_r = gmm;
for k=1:gmm.nmodels
  gmm_r.covs(:,:,k) = regularized_cov(gmm.covs(:,:,k),reg_var,reg_type);
end