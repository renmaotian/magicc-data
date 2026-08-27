# WS11.P — ready-to-paste sentences

## 1. Replacement for the main-text macro-F1 claim

The current sentence reports two point estimates with separate intervals. The replacement reports the **paired** difference, and says plainly where an interval includes zero.

> On the three sets where the three-class comparison is informative, the **paired** cluster-bootstrap difference in MIMAG-inspired macro F1 (MAGICC minus comparator, the same reference-genome clusters resampled for both tools, 2,000 resamples) is: Set C-clean (1,000 genomes in 100 reference clusters): CheckM2 +0.429 [0.353, 0.507]; CoCoPyE +0.277 [0.208, 0.340]; DeepCheck +0.449 [0.370, 0.518] — Set D-clean (1,000 genomes in 100 reference clusters): CheckM2 +0.406 [0.316, 0.485]; CoCoPyE +0.032 [-0.076, 0.131]; DeepCheck +0.567 [0.474, 0.647] — Set E (1,000 genomes in 785 reference clusters): CheckM2 +0.089 [0.053, 0.125]; CoCoPyE +0.015 [-0.020, 0.055]; DeepCheck +0.208 [0.170, 0.247].

> Every interval excludes zero except set_D_clean vs CoCoPyE, set_E vs CoCoPyE, where the paired difference is not distinguishable from zero, so MAGICC's advantage there is not established.

## 2. Ready-to-paste sentence for the 44.2% false-fail rate (register W23)

> At the MIMAG-inspired 5% contamination boundary on Set C-clean, MAGICC rejects 23 of the 52 genomes whose true contamination is below 5% — a false-fail rate of 0.442 [0.273, 0.617], an interval 0.344 wide because the denominator is 52 genomes in 40 reference clusters and not the 1,000 genomes of the set; the companion false-pass rate is 0.020 [0.013, 0.028] on the 948 truly contaminated genomes, and MAGICC's balanced accuracy of 0.769 [0.682, 0.854] is the best of the four tools.

## 3. Ready-to-paste sentence for the ground-truthed anchor

> On the ground-truthed anchor — the 30 of 1,000 Set C-clean genomes that are truly high quality (completeness ≥ 90% and contamination < 5%), spanning 25 reference clusters — MAGICC under-calls completeness by -8.68 pp [-14.17, -3.74] (interval width 10.42 pp) and over-calls contamination by +5.09 pp [1.80, 8.69] (width 6.89 pp), while CheckM2 is near truth (-0.70 and -1.16 pp); with only 30 genomes in 25 clusters the direction is established but the magnitude is not resolved to better than about ±5 pp.

