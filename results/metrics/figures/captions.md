# Figure captions (WS5.6 / WS5.7)

Every caption below states the denominator used for each percentage, as required by Reviewer 1 (major comment 5). Every figure is saved as both PNG (300 dpi) and PDF (vector, Type-42 fonts).

### fig_ws5.3_signed_error_by_set

Signed error distributions (predicted minus true, percentage points) for completeness (top row) and contamination (bottom row), one column per benchmark set, one violin per tool. Violins are kernel density estimates over all genomes of the set (n = 1,000 for Sets A, B, C, D, E, C-clean and D-clean); the inset box shows the median and interquartile range with whiskers at 1.5x IQR; the filled marker is the mean. The dashed line marks zero error: mass above it indicates systematic OVERestimation, below it systematic UNDERestimation. Sets labelled SUPERSEDED were built with dominant genomes drawn from the training split and are shown only for transparency. Colours come from a colour-vision-deficiency-safe palette (no red/green pair; minimum CIE76 Delta-E of 24.6 between any two tool colours under normal, deuteranopic, protanopic and tritanopic vision) and marker shape is a redundant cue. The y-axis is limited to the pooled 0.5-99.5 percentile range so that the informative part of every distribution is legible; a small number of extreme values (chiefly DeepCheck contamination overestimates above +100 pp) fall outside the plotted range and are given in full in ws5.3_signed_errors_overall.tsv.

Denominators, stated explicitly per Reviewer 1 (major comment 5): completeness (%) = retained dominant-genome bp / full reference length of the dominant genome x 100; contamination (%) = total contaminant bp / full reference length of the dominant genome x 100. Both percentages use the same denominator (the full reference length of the dominant genome) and are therefore independent measures; contamination is not normalised to total assembly length or to retained dominant length.

### fig_ws5.3_signed_error_by_contamination_bin

Contamination signed error (predicted minus true, percentage points) stratified by TRUE contamination level, one panel per benchmark set. Reviewer 2 (major comment 2) notes that genomes with 20-80% contamination would be discarded regardless of the exact estimate; this figure therefore resolves the practically relevant low-contamination bins separately from the extreme ones. Violin = kernel density of all genomes in the bin, box = median and IQR (whiskers 1.5x IQR), filled marker = mean; groups with fewer than 10 genomes are drawn as individual points with a median bar. n above each group is the number of genomes in that bin. Negative values indicate UNDERestimation of contamination.

Denominators, stated explicitly per Reviewer 1 (major comment 5): completeness (%) = retained dominant-genome bp / full reference length of the dominant genome x 100; contamination (%) = total contaminant bp / full reference length of the dominant genome x 100. Both percentages use the same denominator (the full reference length of the dominant genome) and are therefore independent measures; contamination is not normalised to total assembly length or to retained dominant length.

### fig_ws5.3_signed_error_by_mimag_class

Contamination signed error stratified by the TRUE MIMAG-inspired quality class of the genome, one panel per benchmark set. Values are contamination signed errors (predicted minus true, percentage points); the dashed line is zero error. Violin = kernel density, box = median and interquartile range (whiskers 1.5x IQR), filled marker = mean; strata with fewer than 10 genomes are drawn as individual points with a median bar, per editorial policy. n above each group is the number of genomes in that stratum.

Denominators, stated explicitly per Reviewer 1 (major comment 5): completeness (%) = retained dominant-genome bp / full reference length of the dominant genome x 100; contamination (%) = total contaminant bp / full reference length of the dominant genome x 100. Both percentages use the same denominator (the full reference length of the dominant genome) and are therefore independent measures; contamination is not normalised to total assembly length or to retained dominant length.

### fig_ws5.3_signed_error_by_contamination_relatedness

Contamination signed error stratified by the taxonomic relatedness between the contaminant genomes and the dominant genome, one panel per benchmark set. `none (uncontaminated)` marks genomes with zero true contamination. Values are contamination signed errors (predicted minus true, percentage points); the dashed line is zero error. Violin = kernel density, box = median and interquartile range (whiskers 1.5x IQR), filled marker = mean; strata with fewer than 10 genomes are drawn as individual points with a median bar, per editorial policy. n above each group is the number of genomes in that stratum.

Denominators, stated explicitly per Reviewer 1 (major comment 5): completeness (%) = retained dominant-genome bp / full reference length of the dominant genome x 100; contamination (%) = total contaminant bp / full reference length of the dominant genome x 100. Both percentages use the same denominator (the full reference length of the dominant genome) and are therefore independent measures; contamination is not normalised to total assembly length or to retained dominant length.

### fig_ws5.3_signed_error_by_phylum

Contamination signed error stratified by the phylum of the dominant genome (phyla with fewer than 30 genomes in a set are pooled as 'Other'), one panel per benchmark set. Values are contamination signed errors (predicted minus true, percentage points); the dashed line is zero error. Violin = kernel density, box = median and interquartile range (whiskers 1.5x IQR), filled marker = mean; strata with fewer than 10 genomes are drawn as individual points with a median bar, per editorial policy. n above each group is the number of genomes in that stratum.

Denominators, stated explicitly per Reviewer 1 (major comment 5): completeness (%) = retained dominant-genome bp / full reference length of the dominant genome x 100; contamination (%) = total contaminant bp / full reference length of the dominant genome x 100. Both percentages use the same denominator (the full reference length of the dominant genome) and are therefore independent measures; contamination is not normalised to total assembly length or to retained dominant length.

### fig_ws5.3_completeness_signed_error_by_mimag_class

Completeness signed error stratified by the TRUE MIMAG-inspired quality class, one panel per benchmark set. Values are completeness signed errors (predicted minus true, percentage points); the dashed line is zero error. Violin = kernel density, box = median and interquartile range (whiskers 1.5x IQR), filled marker = mean; strata with fewer than 10 genomes are drawn as individual points with a median bar, per editorial policy. n above each group is the number of genomes in that stratum.

Denominators, stated explicitly per Reviewer 1 (major comment 5): completeness (%) = retained dominant-genome bp / full reference length of the dominant genome x 100; contamination (%) = total contaminant bp / full reference length of the dominant genome x 100. Both percentages use the same denominator (the full reference length of the dominant genome) and are therefore independent measures; contamination is not normalised to total assembly length or to retained dominant length.

### fig_ws5.1_mimag_macro_f1

Macro-averaged F1 for the MIMAG-inspired 3-class quality assignment (high: completeness >= 90% AND contamination < 5%; medium: completeness >= 50% AND contamination < 10%; low: all others), reported PER BENCHMARK SET rather than as a range. Filled marker = point estimate; whiskers = percentile 95% confidence interval; the shaded violin behind each marker is the full cluster-bootstrap distribution (clusters = dominant reference genomes), which is what replaces a bar chart here. The average is taken over the quality classes that actually occur in the ground truth of each set (Set A contains no low-quality genomes by design), so it is not deflated by structurally empty classes. Note that the full MIMAG standard additionally requires rRNA and tRNA criteria that cannot be evaluated from completeness and contamination estimates.

Denominators, stated explicitly per Reviewer 1 (major comment 5): completeness (%) = retained dominant-genome bp / full reference length of the dominant genome x 100; contamination (%) = total contaminant bp / full reference length of the dominant genome x 100. Both percentages use the same denominator (the full reference length of the dominant genome) and are therefore independent measures; contamination is not normalised to total assembly length or to retained dominant length.

### fig_ws5.1_mimag_confusion_matrices

Confusion matrices for the MIMAG-inspired quality assignment, one panel per benchmark set (rows) and tool (columns). Rows are the true class, columns the predicted class; H = high, M = medium, L = low. Cell shading is the fraction of the true class (row-normalised, so rows sum to 1) on the perceptually uniform, colour-vision-deficiency-optimised `cividis` scale; the printed number is the genome count. Off-diagonal cells above the diagonal are quality DOWNgrades, below the diagonal are UPgrades; the lower-left cell of each matrix therefore counts low-quality genomes wrongly promoted to high quality, the single most consequential error for genome curation.

Denominators, stated explicitly per Reviewer 1 (major comment 5): completeness (%) = retained dominant-genome bp / full reference length of the dominant genome x 100; contamination (%) = total contaminant bp / full reference length of the dominant genome x 100. Both percentages use the same denominator (the full reference length of the dominant genome) and are therefore independent measures; contamination is not normalised to total assembly length or to retained dominant length.

### fig_ws5.2_threshold_error_rates

QC decision errors at the thresholds that matter in practice. A genome PASSES the contamination criterion when contamination < tau and the completeness criterion when completeness >= tau. Top row: false-PASS rate = n(truly fails AND predicted passes) / n(truly fails), i.e. the fraction of genomes that should have been discarded but were retained (equals 1 - sensitivity). Bottom row: false-FAIL rate = n(truly passes AND predicted fails) / n(truly passes), i.e. usable genomes wrongly discarded (equals 1 - specificity). Filled markers are point estimates; whiskers are percentile 95% confidence intervals from 2,000 bootstrap replicates with clusters resampled by dominant reference genome; the shaded violin is the bootstrap distribution. Panels where one class is empty by design (e.g. no contaminated genomes in Set A) are blank.

Denominators, stated explicitly per Reviewer 1 (major comment 5): completeness (%) = retained dominant-genome bp / full reference length of the dominant genome x 100; contamination (%) = total contaminant bp / full reference length of the dominant genome x 100. Both percentages use the same denominator (the full reference length of the dominant genome) and are therefore independent measures; contamination is not normalised to total assembly length or to retained dominant length.

### fig_ws5.5_mae_with_confidence_intervals

Mean absolute error per benchmark set and tool, replacing the bar chart of the original submission. Filled markers are point estimates; whiskers are percentile 95% confidence intervals from 2,000 bootstrap replicates with clusters resampled by dominant reference genome (so that multiple simulations from one reference genome are resampled together). MAE is a single number per set and tool and therefore has no per-genome distribution of its own; the distribution of the underlying per-genome errors is shown in fig_ws5.3_signed_error_by_set and fig_ws5.3_signed_error_by_contamination_bin.

Denominators, stated explicitly per Reviewer 1 (major comment 5): completeness (%) = retained dominant-genome bp / full reference length of the dominant genome x 100; contamination (%) = total contaminant bp / full reference length of the dominant genome x 100. Both percentages use the same denominator (the full reference length of the dominant genome) and are therefore independent measures; contamination is not normalised to total assembly length or to retained dominant length.

### fig_ws5.4_effect_sizes_cliffs_delta

Effect sizes for every MAGICC v5 vs comparator test that entered the Benjamini-Hochberg families. Cliff's delta compares the two absolute-error distributions: delta < 0 means MAGICC v5's errors are stochastically smaller. Horizontal bars are percentile 95% confidence intervals from 2,000 cluster-bootstrap replicates (clusters = dominant reference genomes). Filled markers indicate comparisons that remain significant at q < 0.05 after BH correction under ALL THREE tests (genome-level two-sided Wilcoxon signed-rank, reference-genome-level two-sided Wilcoxon signed-rank on cluster means, and the cluster bootstrap of the difference in MAE); open markers do not. Faint vertical guides mark the conventional |delta| magnitude boundaries 0.147 (negligible/small), 0.33 (small/medium) and 0.474 (medium/large).

Denominators, stated explicitly per Reviewer 1 (major comment 5): completeness (%) = retained dominant-genome bp / full reference length of the dominant genome x 100; contamination (%) = total contaminant bp / full reference length of the dominant genome x 100. Both percentages use the same denominator (the full reference length of the dominant genome) and are therefore independent measures; contamination is not normalised to total assembly length or to retained dominant length.

### fig_ws5.6_palette_cvd_check

Verification that the figure palette satisfies the editorial colour-vision-deficiency requirement. Each row is one palette entry, rendered as seen under normal trichromatic vision and under simulated dichromatic vision (Machado, Oliveira & Fernandes 2009 transformation matrices at severity 1.0, applied in linear RGB). No green is used anywhere in the palette, so no figure requires red/green discrimination. The 6 tool colours were selected by exhaustive search over all 6-colour subsets of the Okabe-Ito palette to maximise the minimum CIE76 Delta-E across all four vision types.

Denominators, stated explicitly per Reviewer 1 (major comment 5): completeness (%) = retained dominant-genome bp / full reference length of the dominant genome x 100; contamination (%) = total contaminant bp / full reference length of the dominant genome x 100. Both percentages use the same denominator (the full reference length of the dominant genome) and are therefore independent measures; contamination is not normalised to total assembly length or to retained dominant length.

