% Construct a GMM.
%
% gmm = gmm_construct(means,covs,mixweights)
function gmm = gmm_construct(means,covs,mixweights)

gmm.dim = size(means,1);
gmm.nmodels = size(means,2);
gmm.means = means;
gmm.covs = covs;
gmm.mixweights = mixweights;
gmm = gmm_fix(gmm);