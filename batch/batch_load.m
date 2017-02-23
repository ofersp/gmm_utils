% Load all samples in batch file. 
%
% batch = batch_load(batch_header_fn)
function batch = batch_load(batch_header_fn)

sample_fn = [batch_header_fn(1:end-11),'_samples.bin'];
batch.header = load(batch_header_fn);
h = batch.header;
batch.properties.sample_bytes = h.sample_dim * class_bytes(h.sample_class);
fp = fopen(sample_fn,'rb');
batch.properties.file_length = file_length(fp);
assert(mod(batch.properties.file_length,batch.properties.sample_bytes) == 0);
batch.properties.n_total_samples = batch.properties.file_length/batch.properties.sample_bytes;

batch.samples = fread(fp,[h.sample_dim,batch.properties.n_total_samples],...
  [h.sample_class,'=>','float64'])*h.sample_scale;
batch.sample_inds = int32(1:batch.properties.n_total_samples);
fclose(fp);
