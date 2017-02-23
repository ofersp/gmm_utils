% Symmetrize GMM covariance matrices.
%
% gmm = gmm_symmetrize(gmm)
function gmm = gmm_symmetrize(gmm)

for i=1:size(gmm.covs,3)
  gmm.covs(:,:,i) = (gmm.covs(:,:,i)+gmm.covs(:,:,i)')/2;
end
gmm = gmm_fix(gmm);
