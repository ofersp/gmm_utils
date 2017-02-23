% Compute weighted average of GMM means
%
% mu = gmm_mean(gmm)

function mu = gmm_mean(gmm)

mu = sum(bsxfun(@times,gmm.mixweights',gmm.means),2);