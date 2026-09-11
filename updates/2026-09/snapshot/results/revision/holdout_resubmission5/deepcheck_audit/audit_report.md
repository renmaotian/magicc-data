# Bounded DeepCheck inference audit

The extreme archived predictions reproduce from the released checkpoint under
the project's documented compatibility adapter. No benchmark predictions were
changed or clipped. This is a bounded numerical and source audit, not a full
rerun or a reconstruction of the authors' original training execution.

## Source, checkpoint and intended output heads

The vendored repository is `https://github.com/Guowei-nju/DeepCheck`, commit
`8453a31c3aeb6282fae246cd67245225d2c9ee5e`. The audit verifies the tracked source,
notebook, scaler, feature names and checkpoint against that exact Git commit
(including the Git LFS object hash where applicable).

* `model.py:77–78,110–112` defines two linear heads: `fc1` is assigned to
  `x_comp`, and `fc2` to `x_cont`. The checked-in forward method computes both
  but returns only `x_comp`.
* `train.py:34–35` divides completeness and contamination labels by 100;
  lines 109 and 129–134 instantiate this ResNet, unpack completeness followed
  by contamination, and assign the corresponding supervised losses. The
  official inference notebook likewise expects both outputs in that order and
  multiplies each by 100. `multi_train.py` follows the same head order.
* The released checkpoint loads strictly into the native ResNet and contains
  both `fc1` and `fc2` weights and biases. Its SHA-256 is
  `5382fcf18e52fa353de7250181b2853e0ca6b77d93add233a651bf8fb6f9a1ea`.
  The source and saved tensors establish the intended contamination head;
  the inconsistency in the released return statement prevents claiming that
  the exact checked-in training program ran unchanged to produce that model.

The minimal return adapter is:

```python
x_comp = self.fc1(x)
x_cont = self.fc2(x)
return x_comp, x_cont  # upstream computes both but returns only x_comp
```

The upstream `model.py` SHA-256 is
`f556d9da8a3e1a4e387c9738e23ae9f001dcea05d8248620d7d3058afb1e8158`;
`train.py` is
`d407ab139cb5d15664538f94f46223e2e6a1c9b2fe797bb7997f53f906f5d8e7`;
the official prediction notebook is
`f000043a8bf3719ee6b9f23db310a534f1c24d95ac66c9c07e772017aa340e02`.
All remaining input and implementation hashes, including the compatibility
adapter, timing code and archived predictions, are in `provenance.tsv`.

## Feature preprocessing and output units

All 48 CheckM2 pickle files examined, comprising 11,820 rows over ten benchmark
sets, contain `Name` followed by exactly the 21,241 features in the released
`feature_names.pkl`, in the same order. The saved `MinMaxScaler` was produced
with scikit-learn 0.23.2. The exported `scale` and `min_val` arrays are exactly
equal to its `scale_` and `min_`; the project's affine transformation is
bitwise equal to the original scaler transform on the eight probe inputs.
The old scaler has no `clip` setting: its historical affine behavior permits
values beyond the fitted range, consistent with the
[scikit-learn 0.23 documentation](https://scikit-learn.org/0.23/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

The adapter scales all 21,241 features, selects the first 20,021, pads with 143
zeros, and reshapes to one 142 × 142 image channel, matching the model's
feature-image recipe. The upstream training helper appends two label columns
and correctly excludes those labels; its separate `read_feature` helper does
not append labels yet still removes the final two columns. On these correctly
shaped CheckM2 vectors that helper would pass only 21,239 columns to a scaler
expecting 21,241. The adapter therefore retains all named features before
scaling. This and the old-pickle compatibility conversion are disclosed
preprocessing adaptations, not an unmodified official CLI.

Both heads are unbounded linear outputs trained in fraction units. The
documented prediction conversion is multiplication by 100 for both heads,
with no extra inverse transformation and no clipping. Negative and greater
than 100 predictions are thus possible and are preserved for scoring.

## Independent numerical replay

The audit loads the unmodified upstream ResNet and captures its already
computed `fc2` tensor with a forward hook. Both native heads match the project
accuracy adapter and the separately implemented runtime adapter bitwise on
eight selected probes. The maximum deviation from archived predictions is
0.0000611 percentage points, consistent with floating-point execution
differences. The requested extrema are:

| Set and genome | Archived contamination (%) | Native checkpoint replay (%) |
|---|---:|---:|
| motivating_v2/set_C, genome_974 | 317.630920 | 317.630890 |
| set_D_clean, genome_818 | −1.563419 | −1.563420 |

These are eight probe selections (one genome appears in two scope-based
extremum selections), not eight randomly sampled independent tests. All
probes and differences are retained in `bounded_extreme_replay.tsv`.

## Reproduction and timing scope

Run the audit from a checkout with the recorded source/checkpoint and existing
benchmark feature pickles:

```bash
conda run -n magicc2 python scripts/265_audit_deepcheck_inference.py
```

The recorded WS8 timing command is a direct Python invocation of the disclosed
adapter (the actual environment executable is recorded in
`scripts/161_ws8_run_one.sh:159–169`):

```bash
/usr/bin/time -v python scripts/163_ws8_deepcheck_infer.py \
  --features <CheckM2-feature-pickle-directory> \
  --output <predictions.tsv> --threads <thread-count>
```

This includes Python startup/imports, model and feature loading, neural
inference and output writing, but starts from precomputed CheckM2 features.
The separate extraction command is `checkm2 predict --input <fasta-directory>
--output-directory <feature-directory> --threads <thread-count> --extension
.fasta --force --dbg_vectors`, using the recorded CheckM2 database and
environment. Complete FASTA-to-result cost includes that extraction stage;
inference-only measurements must remain labelled as such. The timing
campaign avoids `conda run` overhead. That does not justify saying that
DeepCheck used no implementation adapter.

`audit_summary.json` records the completion status, strict checkpoint loading,
head tensor hashes and numerical checks. `feature_schema_audit.tsv` records
each feature file and its SHA-256. The full two-arm MAGICC holdout training is
independent of this comparator audit.
