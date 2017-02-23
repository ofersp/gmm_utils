function R = rand_gpu(varargin)
if varargin{end} == true
  R = gpuArray.rand(varargin{1:end-1});
else
  R = rand(varargin{1:end-1});
end