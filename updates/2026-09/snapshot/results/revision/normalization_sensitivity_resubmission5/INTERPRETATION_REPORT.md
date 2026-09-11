# Frozen-model normalization sensitivity

The recorded production normalizer includes unsupervised feature statistics from all 800,000 V4 training, 100,000 validation and 100,000 test rows. This analysis measures the sensitivity of the frozen V5 predictions to alternative inference transforms. It does not retrain V5, remove preprocessing overlap during training, or estimate the causal effect of that overlap. The benchmark reference split and the preprocessing disclosure therefore remain necessary.

## Inputs and recovery of training-only transforms

The legacy normalizer is `data/features/normalization_params.json` (SHA-256 `b1e3f211a43560d8bed1dd8264921c7ec8ce0529c9bdbdbdc000894c9ce1d7d9`). Its recorded k-mer count is exactly 1,000,000, its standard deviation uses the sample denominator N−1, and all 9,249 k-mer standard deviations are positive with no fallback. The source is `data/features/magicc_features.h5`, with 800,000/100,000/100,000 normalized train/validation/test rows. The source file's size and modification time are recorded; this large HDF5 was not hashed in full.

Before float32 HDF5 storage, the full normalized k-mer data have column sums zero and squared sums 999,999. We read the 200,000 validation/test rows and subtract their column sums and squared sums from these full-data quantities. For each column, if s and q are the held-out sum and squared sum, the recovered training mean is −s/800,000 and its sample variance is `(999999 − q − 800000 × mean²) / 799999`. This saves reading the 800,000 training k-mer rows. Float32 storage introduces small rounding error; the method is not claimed to recover pre-storage statistics bit for bit. As a normalizer fingerprint check, inverse-transforming the first 32 training rows recovers integer raw counts with maximum deviation 0.000401 count.

For the seven summary columns, all 800,000 training rows were read directly. Minima, maxima, medians and interquartile ranges were calculated in their already transformed coordinates. Positive affine transformations preserve these quantities up to the corresponding shift and scale. All passthrough and logarithmic summary transforms remain unchanged. The recovered k-mer mean shifts are at most 0.0088772 in old standardized units, and scale ratios range from 0.995386 to 1.004958.

Three prediction arms use the same released V5 ONNX weights (SHA-256 `b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`):

1. `released_baseline`: the recorded production transforms.
2. `kmer_minmax_training_only`: training-only k-mer mean/standard deviation and summary min/max, retaining the historical robust summary transforms.
3. `all_summary_exact_training`: additionally training-only exact summary medians/interquartile ranges. The historical robust estimates used an unseeded reservoir, so this arm changes both the data used and the quantile estimation procedure. It does not isolate held-out data exclusion alone.

## Predictions and uncertainty

Production k-mer and summary features were freshly counted from all 5,000 FASTAs in Sets A, B, C-clean, D-clean and E. The baseline reproduces the saved V5 predictions with maximum difference 0.000003814 percentage points. For each alternative transform, paired absolute-error changes were calculated on identical assemblies. Their 95% intervals use 2,000 percentile bootstrap resamples of dominant references within each set, preserving all assemblies from a selected reference and retaining sample weighting. There is no retraining or training-seed interval in this analysis.

The k-mer/minmax arm changes set-by-output MAE by −0.017662 to +0.013194 percentage points; the arm that also refits exact robust summaries changes MAE by −0.018979 to +0.011543 percentage points. The largest individual prediction shifts are 0.803158 and 1.460175 percentage points, respectively. The small aggregate shifts support limited sensitivity to these specific inference-time changes on the five-set panel. They do not establish complete preprocessing independence or bound the effect of fitting and training an entirely new model.

All 30 set × arm × output rows, including bias, paired MAE changes and intervals, and median/maximum absolute prediction shifts, are in `normalization_sensitivity.tsv`. Every paired prediction and its truth is in `per_sample_sensitivity.tsv.gz`. Classification results use all three MIMAG-inspired quality labels, including zero F1 for an absent class, and preserve the exact ≥90/≥50 completeness and <5/<10 contamination boundaries. False-fail and false-pass rates at 5% use the truly clean and truly contaminated denominators respectively; balanced accuracy is undefined when either truth class is absent. Full class confusion matrices and all samples whose calls change are retained.

Across both perturbations, no MIMAG-inspired quality class changes in Sets A, B, C-clean or D-clean. Both change three Set E assignments, increasing its three-class macro F1 from 0.903182 to 0.907605. The k-mer/minmax arm changes four 5% contamination calls (two in D-clean, two in E); the exact-summary arm changes two (one in D-clean, one in E). Set C-clean remains at 23/52 false fails and 19/948 false passes, with balanced accuracy 0.768825. D-clean balanced accuracy changes from 0.921556 to 0.937685 or 0.929620; Set E changes from 0.972785 to 0.974100 or 0.972116, respectively. These are descriptive fixed-data sensitivity comparisons. The complete confusion matrices and changed-sample list retain the small adverse as well as beneficial shifts.

## Independent checks and reproduction

`scripts/250_audit_normalization_sensitivity.py` reconstructs new normalizer parameters in raw feature units and applies the production normalization API directly. It compares those results against transformation of the old standardized features for every feature of all 5,000 assemblies. The maximum difference is 1.42 × 10⁻¹⁴ in float64 arithmetic. This separately checks the affine transformation formula and parameter mapping; it does not eliminate the float32 roundoff caveat in recovery of the original training statistics. It also independently recomputes baseline macro F1 and 5% false-call counts and requires agreement with the published benchmark tables before calculating sensitivity.

Reproduce the original analysis with:

```bash
python scripts/249_normalization_sensitivity.py
python scripts/250_audit_normalization_sensitivity.py
```

For archive replay using the recorded affine parameters, without requiring the original V4 HDF5:

```bash
python scripts/249_normalization_sensitivity.py --from-recorded-parameters
python scripts/250_audit_normalization_sensitivity.py
```

The explicit replay flag skips only the original HDF5 identity/moment requirement, records that provenance mode, and still checks released model/normalizer hashes and baseline prediction agreement. When no raw-feature caches exist, it freshly counts the public benchmark FASTAs. The supplied parameter artifact allows reproduction of this recorded sensitivity even though the original unseeded reservoir cannot be recreated exactly. `input_output_manifest.tsv` records SHA-256 hashes of scripts, model, normalizer, vocabulary, benchmark metadata/saved predictions, cached recounted features and numerical outputs. The cache files are derived artifacts; deleting them forces fresh counting.
