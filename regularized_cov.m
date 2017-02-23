% Regularize covariance matrix sigma according to given reg_var and reg_type.
%
% [sigma,log_det_sigma] = regularized_cov(sigma,reg_var,reg_type)
%
% Regularization is performed in one of the following ways: by adding reg_var 
% to the diagonal of covs (when reg_type == 'covs_add'), or by setting all 
% eigen-values of covs below reg_var to reg_var (when reg_type == 'covs_maximize').

function [sigma,log_det_sigma] = regularized_cov(sigma,reg_var,reg_type)

if numel(sigma) == 1  
  if strcmp(reg_type,'covs_add')
    sigma = sigma + reg_var;
    log_det_sigma = log(sigma);
  elseif strcmp(reg_type,'covs_maximize')
    sigma = max(sigma,reg_var);
    log_det_sigma = log(sigma);
  else 
    assert(false); % reg_type unsupported
  end
else
  if strcmp(reg_type,'covs_add')
    sigma = sigma + eye(size(sigma))*reg_var;
    log_det_sigma = log_det(sigma);
  elseif strcmp(reg_type,'covs_maximize')
    [v,d] = eig((sigma'+sigma)/2);
    e = diag(d);
    e = max(reg_var,e);
    d = diag(e);
    sigma = (v*d)/v; 
    sigma = (sigma+sigma')/2;
    log_det_sigma = sum(log(e));  
  else
    assert(false); % reg_type unsupported
  end
end