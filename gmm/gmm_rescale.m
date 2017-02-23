% Rescale the covariances and means of GMM by a factor of alpha.
%
% gmm = gmm_rescale(gmm,alpha)
function gmm = gmm_rescale(gmm,alpha)

gmm.means = gmm.means*alpha;
gmm.covs = gmm.covs*alpha^2;
gmm = gmm_fix(gmm);
