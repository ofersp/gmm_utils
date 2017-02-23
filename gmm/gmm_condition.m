% Condition a GMM on a linear equality constraint.
%
% gmm_c = gmm_condition(gmm,A,y)

function gmm_c = gmm_condition(gmm,A,y)

gmm_c = gmm;
for k=1:gmm.nmodels
    [gmm_c.means(:,k), gmm_c.covs(:,:,k)] = gaussian_condition(...
        gmm.means(:,k), gmm.covs(:,:,k),A,y);
end