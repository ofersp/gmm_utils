% Compute the product GMM of 2 given GMMs (slower version)
%
% gmm3 = gmm_multiply(gmm1,gmm2,normalize)
function gmm3 = gmm_multiply(gmm1,gmm2,normalize)

if ~exist('normalize','var'); normalize = true; end
assert(gmm1.dim == gmm2.dim);

d = gmm1.dim;

gmm1_means = gmm1.means;
gmm1_covs = gmm1.covs;
gmm1_mixweights = gmm1.mixweights;
gmm2_means = gmm2.means;
gmm2_covs = gmm2.covs;
gmm2_mixweights = gmm2.mixweights;

gmm3.dim = d;
gmm3.nmodels = gmm1.nmodels*gmm2.nmodels;
gmm3_means = zeros(d,gmm1.nmodels,gmm2.nmodels);
gmm3_covs = zeros(d,d,gmm1.nmodels,gmm2.nmodels);
gmm3_mixweights = zeros(gmm1.nmodels,gmm2.nmodels);

for i=1:gmm1.nmodels   
  for j=1:gmm2.nmodels
    [mu3,cov3,log_c3] = gaussian_multiply_fast(...
        gmm1_means(:,i),gmm1_covs(:,:,i),gmm2_means(:,j),gmm2_covs(:,:,j));
    gmm3_means(:,i,j) = mu3;
    gmm3_covs(:,:,i,j) = cov3;
    gmm3_mixweights(i,j) = log_c3 + log(gmm1_mixweights(i)) + log(gmm2_mixweights(j));
  end
end

gmm3.means = reshape(gmm3_means,d,gmm3.nmodels);
gmm3.covs = reshape(gmm3_covs,d,d,gmm3.nmodels);
gmm3.mixweights = reshape(gmm3_mixweights,gmm3.nmodels,1);

if normalize
  gmm3.mixweights = gmm3.mixweights - log_sum_exp(gmm3.mixweights); % normalize
  gmm3.mixweights = exp(gmm3.mixweights);
end

gmm3 = gmm_symmetrize(gmm3);