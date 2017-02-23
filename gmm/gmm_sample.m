% Random samples from a GMM distribution.
%
% X = gmm_sample(gmm,n)
% X = gmm_sample(gmm,n,batch_sz)

function X = gmm_sample(gmm,n,batch_sz)

if ~exist('batch_sz','var'); batch_sz = 5000; end
num_full_batches = floor(n/batch_sz);
last_batch_sz = mod(n,batch_sz);

X = zeros(gmm.dim,n);
for i=1:num_full_batches
  X(:,1+(i-1)*batch_sz:i*batch_sz) = gmm_sample_(gmm,batch_sz);
end

if last_batch_sz>0
  X(:,end-last_batch_sz+1:end) = gmm_sample_(gmm,last_batch_sz);
end

function X = gmm_sample_(gmm,n)

models = randsample(1:gmm.nmodels,n,true,double(gmm.mixweights));
X = mvnrnd(gmm.means(:,models)',gmm.covs(:,:,models))';
