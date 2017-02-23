% Compute the MAP estimate of a variable having GMM prior and gaussian innovation. 
%
% X = gmm_map_inference(X0,Y,sigma,gmm,num_iters)
function X = gmm_map_inference(X0,Y,sigma,gmm,num_iters)

X = X0;
noise_var = sigma^2;
% prepare filters and deltas
A = zeros(size(gmm.covs));
deltas = zeros(size(Y,1),gmm.nmodels);

for i=1:gmm.nmodels
    A(:,:,i) = (gmm.covs(:,:,i)+noise_var*eye(size(Y,1)))\gmm.covs(:,:,i);
    deltas(:,i) = (gmm.covs(:,:,i)+noise_var*eye(size(Y,1)))\(noise_var*gmm.means(:,i));
end

% EM itrations
for t=1:num_iters
    % E step
    log_resp_unnorm = calc_resp(X,gmm);
    % M step
    X = hard_filter(Y,A,deltas,log_resp_unnorm,X);
end

function log_resp_unnorm = calc_resp(X,gmm)
% full posterior calculation
log_resp_unnorm = zeros(gmm.nmodels,size(X,2));
for i=1:gmm.nmodels
    log_resp_unnorm(i,:) = log(gmm.mixweights(i)) + ...
        log_gauss_pdf(X,gmm.covs(:,:,i),gmm.means(:,i));
end

function X = hard_filter(Y,A,deltas,log_resp_unnorm,X)
% filter noisy patches according to most probable component 
[~,ks] = max(log_resp_unnorm);
for i=1:size(A,3)
    idx = ks==i;
    X(:,idx) = bsxfun(@plus,A(:,:,i)*Y(:,idx),deltas(:,i));
end
