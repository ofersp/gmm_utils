% Sort GMM components in descending mixweight order
%
% [gmm_sorted,orig_inds] = gmm_sort(gmm_orig,mode)
function [gmm_sorted,orig_inds] = gmm_sort(gmm_orig,mode)

if ~exist('mode','var'); mode = 'descend'; end

[~,orig_inds] = sort(gmm_orig.mixweights,1,mode); 
gmm_sorted = gmm_orig;
gmm_sorted.mixweights = gmm_orig.mixweights(orig_inds);
gmm_sorted.means = gmm_orig.means(:,orig_inds);
gmm_sorted.covs = gmm_orig.covs(:,:,orig_inds);
