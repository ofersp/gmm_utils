% Load a mini-batch of samples from a batch file.
%
% m_batch = batch_load_mini(batch_header_fn,n_mb_samples,mb_ind)
function m_batch = batch_load_mini(batch_header_fn,n_mb_samples,mb_ind)

sample_fn = [batch_header_fn(1:end-11),'_samples.bin'];
m_batch.header = load(batch_header_fn);
h = m_batch.header;
m_batch.properties.sample_bytes = h.sample_dim * class_bytes(h.sample_class);
fp = fopen(sample_fn,'rb');
m_batch.properties.file_length = file_length(fp);
assert(mod(m_batch.properties.file_length,m_batch.properties.sample_bytes) == 0);
m_batch.properties.n_total_samples = m_batch.properties.file_length/m_batch.properties.sample_bytes;

m_batch.properties.mb_bytes = m_batch.properties.sample_bytes*n_mb_samples;
m_batch.properties.n_mb_avail = floor(m_batch.properties.file_length/m_batch.properties.mb_bytes);
assert(m_batch.properties.n_mb_avail >= 1);
if exist('mb_ind','var')
  assert(m_batch.properties.n_mb_avail >= mb_ind); 
else
  mb_ind = randi(m_batch.properties.n_mb_avail);
end

fseek(fp,(mb_ind-1)*m_batch.properties.mb_bytes,'bof');
m_batch.samples = fread(fp,[h.sample_dim,n_mb_samples],...
  [h.sample_class,'=>','float64'])*h.sample_scale;
m_batch.sample_inds = int32(1+(mb_ind-1)*n_mb_samples:mb_ind*n_mb_samples);
fclose(fp);
