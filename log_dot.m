% Compute S = log(exp(A)*exp(B)) in a precision-aware manner.
%
%   Arguments:
%       A - a log space matrix (MxN)
%       B - a log space matrix (NxK)
%       logprec - argmax_s such that exp(-s) >> 0 in this system. i.e.
%                 maximal number such that its negative exponent is not
%                 considered 0 by MATLAB. Defaults to 700.

function S = log_dot(A, B, logprec)
if ~exist('logprec', 'var') || isempty(logprec) logprec = 700; end;
    
%if (min(A(:)) > -logprec/2) && (min(B(:)) > -logprec/2)
if (max(abs(A(:))) < logprec/2) && (max(abs(B(:))) < logprec/2)
    S = log(exp(A)*exp(B));
else
    M = size(A,1);
    K = size(B,2);
    repB = permute(repmat(B, [1, 1, M]), [3,1,2]);
    repA = repmat(A, [1, 1, K]);
    S = logsum(repA+repB, 2);
end
end