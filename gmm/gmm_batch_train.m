% Train a GMM on given dataset using full batch expectation-maximization.
%
% gmm_batch_train(batch_header_fn,n_epochs,nmodels,reg_var,out_dir)
% gmm_batch_train(batch_header_fn,n_epochs,nmodels,reg_var,out_dir,seed)
% gmm_batch_train(batch_header_fn,n_epochs,nmodels,reg_var,out_dir,seed,use_gpu)
%
% TODO: add logging either here or in gmm_ml_training.m

function gmm_batch_train(batch_header_fn,n_epochs,nmodels,reg_var,out_dir,seed,use_gpu)

if nargin() < 7; use_gpu = false; end
if nargin() < 6; seed = 1234; end; rng(seed);

fprintf(1,'Reading training samples. ');
batch = batch_load(batch_header_fn);
fprintf(1,'%d samples read.\n\n',size(batch.samples,2));

for i=1:numel(nmodels)
for j=1:numel(reg_var)
  gmm_batch_train_(batch,n_epochs,nmodels(i),reg_var(j),out_dir,seed,use_gpu);
end
end

function gmm_batch_train_(batch,n_epochs,nmodels,reg_var,out_dir,seed,use_gpu)

fname_fmt = '%s/gmm_nmodels_%d_regvar_%.0e';
fname_base = sprintf(fname_fmt,out_dir,nmodels,reg_var);
fname_model = [fname_base,'.mat'];

gmm0.nmodels = nmodels;
[gmm,ll] = gmm_ml_training(batch.samples,gmm0,n_epochs,reg_var,use_gpu);
avg_l2l_per_dim = mean(ll/log(2))/gmm.dim;
fprintf(1,'nmodels=%d, reg_var=%.0e, l2l_per_dim=%.3e\n',gmm.nmodels,reg_var,avg_l2l_per_dim);

% save the model with its meta-data
gmm = gmm_symmetrize(gmm);
if use_gpu; gmm = structfun2struct(@gather,gmm); end
gmm.metadata.was_online = false;
gmm.metadata.n_epochs = n_epochs;
gmm.metadata.seed = seed;
save(fname_model,'gmm');
