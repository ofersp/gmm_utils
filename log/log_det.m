% Compute log(det(A)) in a precision-aware manner.
%
% ld = log_det(A)
function ld = log_det(A)
e = eig(A);
ld = sum(log(e));
