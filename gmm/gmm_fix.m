% Fix or add auxilary fields of a GMM struct. 
%
% gmm = gmm_fix(gmm)
function gmm = gmm_fix(gmm)


assert(isfield(gmm,'mixweights'));

gmm.nmodels = size(gmm.means,2);
gmm.dim = size(gmm.means,1);
gmm.mixweights = gmm.mixweights(:);
