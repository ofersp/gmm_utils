% Fast multi-variate-normal PDF evaluation when logdetcovs exist.
%
% log_prob = log_mvn_pdf_fast(X,mu,invcov,logdetcov)
function log_prob = log_mvn_pdf_fast(X,mu,invcov,logdetcov)

mu = mu(:)';
d = numel(mu);
X = bsxfun(@plus,X,-mu);
log_prob = -0.5*(d*log(2*pi)+logdetcov+dot(X,X*invcov,2));