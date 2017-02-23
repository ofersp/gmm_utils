% GMM
%
% Files
%   gmm_batch_train        - Train a GMM on given dataset using full batch expectation-maximization.
%   gmm_batch_train_online - Train a GMM on given dataset using mini batch expectation-maximization.
%   gmm_construct          - Construct a GMM.
%   gmm_expand             - Expand a low dimensional GMM to model a higher dimensional sample.
%   gmm_fix                - Fix or add auxilary fields of a GMM struct. 
%   gmm_log_likelihood     - Compute log-likelihood of a GMM at given points.
%   gmm_map_inference      - Compute the MAP estimate of a variable having GMM prior and gaussian innovation. 
%   gmm_ml_training        - Train a gaussian-mixture-model (GMM) using expectation-maximization.
%   gmm_ml_training_demo   - Demo of gmm_ml_training's usage.
%   gmm_multiply           - Compute the product GMM of 2 given GMMs (slower version)
%   gmm_rescale            - Rescale the covariances and means of GMM by a factor of alpha.
%   gmm_sample             - Random samples from a GMM distribution.
%   gmm_sample_comps       - Random samples from specific components of a GMM distribution.
%   gmm_sort               - Sort GMM components in descending mixweight order
%   gmm_symmetrize         - Symmetrize GMM covariance matrices.
%   gmm_weighted_avg       - Compute weighted average GMM of the means and covariances of 2 GMMs.
%   gmm_condition          - Condition a GMM on a linear equality constraint.
%   gmm_mean               - Compute weighted average of GMM means
%   gmm_regularize         - Regularize all of GMM components.
%   gmm_compress           - Reduce the number of components in a GMM.
