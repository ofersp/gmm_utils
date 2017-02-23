% GMM_UTILS
%
% Files
%   gaussian_condition     - Condition a gaussian variable on a linear equality constraint.
%   gaussian_multiply      - Product of gaussian densities.
%   gaussian_multiply_fast - Product of gaussian densities (faster version).
%   gmm_utils_demo         - Train a gmm on in memory samples
%   gmm_utils_setup        - These subdirectories need to be in the matlab path
%   log_det                - Compute log(det(A)) in a precision-aware manner.
%   log_dot                - Compute S = log(exp(A)*exp(B)) in a precision-aware manner.
%   log_gauss_norm_const   - Evaluate the logarithmic normaliztion constant of given gaussian covariance
%   log_gauss_pdf          - Evaluate log-gaussian PDF.
%   log_mvn_pdf_fast       - Fast multi-variate-normal PDF evaluation when logdetcovs exist.
%   log_sum_exp            - Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
%   regularized_cov        - Regularize covariance matrix sigma according to given reg_var and reg_type.
