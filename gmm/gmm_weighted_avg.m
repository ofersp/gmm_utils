% Compute weighted average GMM of the means and covariances of 2 GMMs.
%
% gmm3 = gmm_weighted_avg(gmm1,gmm2,eta,use_gpu)
function gmm3 = gmm_weighted_avg(gmm1,gmm2,eta)

gmm3 = gmm2;
if isfield(gmm1,'mixweights') && isfield(gmm1,'means') && isfield(gmm1,'covs')
  gmm3.mixweights = gmm1.mixweights*(1-eta)+gmm2.mixweights*eta;
  gmm3.means = gmm1.means*(1-eta)+gmm2.means*eta;
  gmm3.covs = gmm1.covs*(1-eta)+gmm2.covs*eta;
  gmm3 = gmm_fix(gmm3);
end

