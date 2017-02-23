% Demo of gmm_ml_training's usage.
function gmm_mle = gmm_ml_training_demo()

clear;
rng('default');

num_iters = 100;
num_samples = 5e4;
mineig = 0.1;
maxeig = 8;
minmean = -2;
maxmean = 2;
plot_lims = [-5,5];
dim = 3;
nmodels = 2;
reg_var = 0.01;
use_gpu = false;
reg_type = 'covs_maximize';
zero_means = false;

gmm = rand_gmm(mineig,maxeig,minmean,maxmean,dim,nmodels);
batch = gmm_sample(gmm,num_samples);
gmm0.nmodels = nmodels;
tic;
gmm_mle = gmm_ml_training(batch,gmm0,num_iters,reg_var,use_gpu,reg_type,zero_means); 
toc;

figure(1);
dscatter(batch(1,:)',batch(2,:)');
axis([minmean,maxmean,minmean,maxmean]); hold on;
plot_gaussian_ellipse(gmm.covs(1:2,1:2,:),gmm.means(1:2,:),exp(1),'r'); 
plot_gaussian_ellipse(gmm_mle.covs(1:2,1:2,:),gmm_mle.means(1:2,:),exp(1),'g'); 
xlim(plot_lims);
ylim(plot_lims);
hold off;

fprintf(1,'log-likelihood of training set using gmm: %5.5e\n',...
  sum(gmm_log_likelihood(batch,gmm)));
fprintf(1,'log-likelihood of training set using mle gmm: %5.5e\n',...
  sum(gmm_log_likelihood(batch,gmm_mle)));

function gmm = rand_gmm(mineig,maxeig,minmean,maxmean,dim,nmodels)

gmm.dim = dim;
gmm.nmodels = nmodels;

gmm.mixweights = ones(gmm.nmodels,1)/gmm.nmodels;
gmm.mixweights = gmm.mixweights/sum(gmm.mixweights);
gmm.covs = rand_psd(gmm.dim,mineig,maxeig,gmm.nmodels);
gmm.means = minmean + (maxmean-minmean)*rand(gmm.dim,gmm.nmodels);

function M = rand_psd(dim,mineig,maxeig,n)

M = zeros(dim,dim,n);
for i=1:n
  A = rand(dim)-0.5;
  A = A+A';
  [~,D] = eig(A); 
  d = diag(D);
  A = A - min(d)*eye(dim);
  A = A*(maxeig-mineig)/(max(d)-min(d));
  A = A + mineig*eye(dim);
  M(:,:,i) = A;
end
