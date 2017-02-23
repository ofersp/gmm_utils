% Random samples from specific components of a GMM distribution.
%
% X = gmm_sample_comps(gmm,n,comp_inds)

function X = gmm_sample_comps(gmm,n,comp_inds)

X = zeros(gmm.dim,n,length(comp_inds));
for i=1:length(comp_inds)
  k = repmat(comp_inds(i),n,1);
  X(:,:,i) = mvnrnd(gmm.means(:,k)',gmm.covs(:,:,k))';
end