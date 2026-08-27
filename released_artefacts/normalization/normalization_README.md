# Normalization parameters (WS7.6)

`normalization_params.json` holds the per-feature mean/std applied at
inference: k-mer counts are transformed with `log10(count + 1)` and then
z-scored with these values; the 7 k-mer summary features are z-scored with
their own stored scalers. Implementation: `magicc/normalization.py`.

These parameters were fitted by streaming Welford statistics over the
**training** synthesis stream only (reservoir sampling for the summary-feature
scalers), so no validation or test genome influenced them.

SHA256 `b1e3f211a43560d8bed1dd8264921c7ec8ce0529c9bdbdbdc000894c9ce1d7d9`

Top-level keys: `assembly_log10_offset`, `assembly_minmax_min`, `assembly_minmax_range`, `assembly_robust_iqr`, `assembly_robust_median`, `assembly_stats`, `finalized`, `kmer_mean`, `kmer_stats`, `kmer_std`, `n_assembly_features`, `n_kmer_features`
