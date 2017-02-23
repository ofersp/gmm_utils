% Shuffle samples in batch file.
%
% batch_shuf(batch_fn,sample_dim,sample_class,seed)
function batch_shuf(batch_fn,sample_dim,sample_class,seed)

if exist('seed','var'); rng(seed); end
fp_inp = fopen(batch_fn,'rb');
batch = fread(fp_inp,[sample_dim,Inf],sample_class);
fclose(fp_inp);
batch = batch(:,randperm(size(batch,2)));
fp_out = fopen(batch_fn,'wb');
fwrite(fp_out,batch,sample_class);
fclose(fp_out);
