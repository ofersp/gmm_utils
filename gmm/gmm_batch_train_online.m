% Train a GMM on given dataset using mini batch expectation-maximization.
%
% gmm_batch_train_online(batch,n_epochs,nmodels,reg_var,n_mb_samples,eta,alpha,out_dir)
% gmm_batch_train_online(batch,n_epochs,nmodels,reg_var,n_mb_samples,eta,alpha,out_dir,seed)
% gmm_batch_train_online(batch,n_epochs,nmodels,reg_var,n_mb_samples,eta,alpha,out_dir,seed,use_gpu)
% gmm_batch_train_online(batch,n_epochs,nmodels,reg_var,n_mb_samples,eta,alpha,out_dir,seed,use_gpu,log_on)

function gmm_batch_train_online(batch,n_epochs,nmodels,reg_var,n_mb_samples,...
  eta,alpha,out_dir,seed,use_gpu,log_on)

if nargin() < 11; log_on = true; end
if nargin() < 10; use_gpu = false; end
if nargin() < 9; seed = 1234; end; rng(seed);
if isempty(eta); eta = 0.3; alpha = NaN; end

if ischar(batch)
  fprintf(1,'Reading training samples. ');
  batch = batch_load(batch);
  fprintf(1,'%d samples read.\n\n',size(batch.samples,2));
end

for i=1:numel(nmodels)
for j=1:numel(reg_var)
for k=1:numel(n_mb_samples)
for l=1:numel(eta) 
for m=1:numel(alpha) 
  gmm_batch_train_online_(batch,n_epochs,nmodels(i),reg_var(j),n_mb_samples(k),...
    eta(l),alpha(m),out_dir,seed,use_gpu,log_on);
end
end
end
end
end

function gmm_batch_train_online_(batch,n_epochs,nmodels,reg_var,n_mb_samples,eta,alpha,out_dir,seed,use_gpu,log_on)

% set output file names
fname_fmt = '%s/gmm_nmodels_%d_regvar_%.0e_mbsz_%d_eta_%.0e_alpha_%.0e';
fname_base = sprintf(fname_fmt,out_dir,nmodels,reg_var,n_mb_samples,eta,alpha);
fname_model = [fname_base,'.mat'];
fname_log = [fname_base,'.log'];
% open the log file
if log_on; fid_log = fopen(fname_log,'w'); end

n_mbs_per_epoch = ceil(size(batch.samples,2)/n_mb_samples);
curr_mb = 1;
gmm = [];
gmm.nmodels = nmodels;

for curr_epoch=1:n_epochs
  for curr_mb_in_epoch=1:n_mbs_per_epoch; tic;
    X = batch.samples(:,randperm(size(batch.samples,2),n_mb_samples));
    d = size(batch.samples,1);

    [gmm_tmp,ll] = gmm_ml_training(X,gmm,1,reg_var,use_gpu);
    if ~isnan(alpha); eta = eta_schedule(alpha,curr_mb-1); end
    gmm = gmm_weighted_avg(gmm,gmm_tmp,eta,use_gpu);
    avg_l2l_per_dim = mean(ll/log(2))/d;

    % print and possibly log training progress
    msg = sprintf(['nmodels=%d, reg_var=%.0e, mb_sz=%d, alpha=%.0e, ',...
      'epoch: %d/%d, epoch-mb: %d/%d, l2l_per_dim=%.3e, eta=%.2e\n'],...
      gmm.nmodels,reg_var,n_mb_samples,alpha,curr_epoch,n_epochs,curr_mb_in_epoch,...
      n_mbs_per_epoch,avg_l2l_per_dim,eta);
    fprintf(1,msg);
    if log_on; fprintf(fid_log,msg); end

    curr_mb = curr_mb + 1; toc;
  end  
end
fprintf(1,'\n');

% save the model with its meta-data
gmm = gmm_symmetrize(gmm);
if use_gpu; gmm = structfun2struct(@gather,gmm); end
gmm.metadata.was_online = true;
gmm.metadata.n_epochs = n_epochs;
gmm.metadata.n_mb_samples = n_mb_samples;
gmm.metadata.n_mbs_per_epoch = n_mbs_per_epoch;
gmm.metadata.seed = seed;
if isnan(alpha); 
  gmm.metadata.eta = eta;
  gmm.metadata.alpha = NaN;
else
  gmm.metadata.eta = NaN;
  gmm.metadata.alpha = alpha;
end
save(fname_model,'gmm');

% close the log file
if log_on; fclose(fid_log); end

function eta = eta_schedule(alpha,k)
% see: 2009 - Liang, Klein - Online EM for Unsupervised Models
eta = (k+2)^(-alpha);
