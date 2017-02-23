% Reduce the number of components in a GMM.
%
% gmm = gmm_compress(gmm,nmodels)
function gmm = gmm_compress(gmm,nmodels)

gmm = gmm_sort(gmm);
gmm.means = gmm.means(:,1:nmodels);
gmm.covs = gmm.covs(:,:,1:nmodels);
gmm.mixweights = gmm.mixweights(1:nmodels);
gmm = gmm_fix(gmm);
