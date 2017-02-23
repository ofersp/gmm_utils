function Z = zeros_gpu(varargin)
if varargin{end} == true
  Z = gpuArray.zeros(varargin{1:end-1});
else
  Z = zeros(varargin{1:end-1});
end