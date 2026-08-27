# WS4.2 — GUNC as a standard detection comparator, with database sensitivity and a statistical-power audit

Generated 2026-08-01T01:37:15.409080+00:00 by `scripts/138_ws42_gunc_comparator.py`. GUNC 1.1.1, DIAMOND 2.1.24, Prodigal V2.6.3.

## 0. How GUNC is (and is not) used here

GUNC returns a **clade separation score (CSS)** and a **pass/fail** flag. It does **not** estimate completeness or contamination as percentages, so it is deliberately **excluded from the MAE table** (protocol §4.4c item 5). It is scored here as a *detector*, and MAGICC V5 and CheckM2 are scored on the same detection task at the same thresholds so the comparison is like-for-like. One asymmetry favours the estimators and is stated rather than hidden: they are thresholded at exactly the MIMAG-inspired level being tested, whereas GUNC has a single built-in operating point (CSS > 0.45 at the maxCSS taxonomic level) that cannot be re-tuned per level. Section 4 (Spearman/AUROC) is threshold-free and removes that asymmetry.

**MIMAG-inspired thresholds** used throughout: high quality ≥90 % completeness AND <5 % contamination; medium quality ≥50 % AND <10 % (Bowers et al. 2017). "MIMAG-inspired" because the strict definition additionally requires rRNA/tRNA criteria that cannot be evaluated from these assemblies. The ground-truth positive (contaminated) class is therefore *true contamination ≥ threshold*.

### Database choice is a methodological decision, and the trade-off is real

| Database | reference genomes | phyla | CPR/Patescibacteria refs | archaeal refs |
|---|---|---|---|---|
| proGenomes 2.1 | 11,223 | 149 | **104** | 457 |
| GTDB r95 | 31,910 | 149 | **1,131** | 1,672 |

proGenomes 2.1 holds only **104** CPR references, so GUNC is expected to be near-blind on `set_C_clean`, which is entirely Patescibacteriota. GTDB databases give far better novel-lineage coverage **but include MAG-derived reference genomes** (14,566 GenBank-prefixed entries in r95), i.e. reference genomes that may themselves be contaminated. proGenomes 2.1 is the cleaner reference set but blind to CPR/DPANN. **Neither is strictly superior**; both are run and the comparison is reported as-is.

### The power caveat, stated precisely

GUNC computes `genes_retained_index (GRI) = genes retained in abundant clades / genes called` and `reference_representation_score (RRS) = GRI × mean AA hit identity`. When **GRI ≤ 0.4 the CSS is forcibly multiplied by zero**, so the genome **automatically passes** no matter how chimeric it actually is. A pass from such a genome is *not* evidence of cleanliness — it is evidence that GUNC had no close reference. Earlier control testing on novel-lineage MAGs found 7/8 positives with RRS < 0.5 and GUNC flagged 0/8, while correctly passing 6/6 finished pure cultures. Power strata used below:

* `hard_unpowered` — GRI ≤ 0.4 (CSS forcibly zeroed)
* `weak` — GRI > 0.4 but RRS < 0.5
* `powered` — GRI > 0.4 and RRS ≥ 0.5

`unpowered` = `hard_unpowered` ∪ `weak`. GUNC's own caution flag (RRS < 0.3) is reported separately.


## 0b. Headline findings

**1. Powered fraction — the number that gates every other number here.** set_A_v2/proGenomes 2.1 **98.4 %** (984/1000); set_B_v2/proGenomes 2.1 **99.3 %** (993/1000); set_C_clean/proGenomes 2.1 **68.1 %** (681/1000); set_C_clean/GTDB r95 **88.0 %** (880/1000); set_D_clean/proGenomes 2.1 **94.5 %** (945/1000); set_D_clean/GTDB r95 **97.4 %** (974/1000); set_E/proGenomes 2.1 **98.6 %** (986/1000); set_E/GTDB r95 **99.2 %** (992/1000).

The two novel-lineage clean sets are where GUNC is least able to adjudicate, and `set_C_clean` (entirely Patescibacteriota) under proGenomes 2.1 is the worst case. Nothing in the `unpowered` stratum should be read as evidence of cleanliness.


**2. GUNC's false-positive rate is a property of novel lineages, not of GUNC.** Fail rate on genomes whose *true* contamination is < 5 % — i.e. genomes GUNC should pass:

| Set | Database | n clean | fail rate (all strata) | fail rate (powered) |
|---|---|---|---|---|
| set_A_v2 | proGenomes 2.1 | 1000 | **0.013** | 0.013 |
| set_B_v2 | proGenomes 2.1 | 200 | **0.030** | 0.030 |
| set_C_clean | proGenomes 2.1 | 52 | **0.173** | 0.267 |
| set_C_clean | GTDB r95 | 52 | **0.423** | 0.452 |
| set_D_clean | proGenomes 2.1 | 62 | **0.629** | 0.684 |
| set_D_clean | GTDB r95 | 62 | **0.645** | 0.678 |
| set_E | proGenomes 2.1 | 252 | **0.171** | 0.172 |
| set_E | GTDB r95 | 252 | **0.175** | 0.175 |

On `set_A_v2` — 1,000 uncontaminated genomes spanning 31 phyla of mainstream taxa — GUNC is essentially never wrong. On the CPR and archaeal clean sets it is wrong constantly. **The failure mode is reference representation, and it is invisible unless the strata are reported separately.**

⚠️ **Stratum discipline.** On `set_D_clean` under proGenomes 2.1 GUNC fails a large majority of genuinely clean archaeal genomes: **68.4 % in the powered stratum, 62.9 % across all strata**. These two numbers are not interchangeable and neither may be quoted without its stratum. The same counter-finding holds under GTDB r95 (67.8 % powered, 64.5 % all strata), so it is not a database artefact.


**3. Rank correlation with true contamination — the three tools side by side** (stratum `all`; full table with CIs in §4):

| Set | Database | GUNC CSS | GUNC contamination_portion | MAGICC V5 | CheckM2 | GUNC CSS AUROC ≥5 % |
|---|---|---|---|---|---|---|
| set_B_v2 | proGenomes 2.1 | 0.572 | 0.925 | 0.959 | 0.794 | 0.999 |
| set_C_clean | proGenomes 2.1 | 0.321 | 0.366 | 0.891 | 0.610 | 0.889 |
| set_C_clean | GTDB r95 | 0.200 | 0.847 | 0.891 | 0.610 | 0.832 |
| set_D_clean | proGenomes 2.1 | 0.103 | 0.788 | 0.933 | 0.693 | 0.703 |
| set_D_clean | GTDB r95 | 0.199 | 0.858 | 0.933 | 0.693 | 0.726 |
| set_E | proGenomes 2.1 | 0.559 | 0.837 | 0.960 | 0.846 | 0.913 |
| set_E | GTDB r95 | 0.622 | 0.863 | 0.960 | 0.846 | 0.923 |

MAGICC V5 ranks contamination best on every set. GUNC's CSS ranks worst — but the AUROC column shows why that is **saturation, not a failure of detection**: CSS pins at 1.0 as soon as a genome is confidently chimeric, so it carries little rank information among contaminated genomes while still separating them from clean ones. GUNC's `contamination_portion` ranks far better than its CSS. This is exactly why GUNC is scored as a detector and **kept out of the MAE table**.


**4. Where GUNC beats the estimators, stated plainly.** GUNC out-detects CheckM2 on recall at ≥5 % on every set tested — most starkly on the novel-lineage sets, where CheckM2 recovers roughly half (`set_C_clean`) to two-thirds (`set_D_clean`) of contaminated genomes while GUNC recovers nearly all of them. CheckM2 buys that with perfect specificity on those sets; GUNC does not. Both facts belong in the record.


## 1. Power audit — how much of each set GUNC could actually adjudicate

| Set | Database | n | scored | **powered** | **% powered** | hard-unpowered (CSS zeroed) | weak (RRS<0.5) | RRS<0.3 | median RRS | median GRI | median AA id | median genes mapped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| set_A_v2 | proGenomes 2.1 | 1000 | 1000 | **984** | **98.4 %** | 0 | 16 | 2 | 0.920 | 0.980 | 0.960 | 2507 |
| set_B_v2 | proGenomes 2.1 | 1000 | 1000 | **993** | **99.3 %** | 0 | 7 | 0 | 0.900 | 0.960 | 0.940 | 4606 |
| set_C_clean | proGenomes 2.1 | 1000 | 1000 | **681** | **68.1 %** | 1 | 318 | 12 | 0.560 | 0.740 | 0.770 | 992 |
| set_C_clean | GTDB r95 | 1000 | 1000 | **880** | **88.0 %** | 0 | 120 | 2 | 0.640 | 0.830 | 0.800 | 1022 |
| set_D_clean | proGenomes 2.1 | 1000 | 1000 | **945** | **94.5 %** | 0 | 55 | 13 | 0.840 | 0.940 | 0.890 | 2470 |
| set_D_clean | GTDB r95 | 1000 | 1000 | **974** | **97.4 %** | 1 | 25 | 1 | 0.920 | 0.970 | 0.950 | 2516 |
| set_E | proGenomes 2.1 | 1000 | 1000 | **986** | **98.6 %** | 0 | 14 | 4 | 0.900 | 0.960 | 0.950 | 4012 |
| set_E | GTDB r95 | 1000 | 1000 | **992** | **99.2 %** | 0 | 8 | 1 | 0.940 | 0.970 | 0.970 | 4044 |

Fail rates by stratum (a "pass" in the unpowered stratum is uninformative):

| Set | Database | overall fail rate | powered fail rate | unpowered fail rate |
|---|---|---|---|---|
| set_A_v2 | proGenomes 2.1 | 0.013 | 0.013 | 0.000 |
| set_B_v2 | proGenomes 2.1 | 0.805 | 0.808 | 0.429 |
| set_C_clean | proGenomes 2.1 | 0.862 | 0.955 | 0.665 |
| set_C_clean | GTDB r95 | 0.969 | 0.980 | 0.892 |
| set_D_clean | proGenomes 2.1 | 0.971 | 0.981 | 0.800 |
| set_D_clean | GTDB r95 | 0.978 | 0.981 | 0.885 |
| set_E | proGenomes 2.1 | 0.761 | 0.767 | 0.357 |
| set_E | GTDB r95 | 0.772 | 0.775 | 0.375 |

## 2. GUNC pass/fail stratified by true contamination

Bands are `[lo, hi)` in percent contamination.


### 2.1 stratum = `all`

| Set | Database | band (%) | n | scored | fail | fail rate | median CSS | median RRS |
|---|---|---|---|---|---|---|---|---|
| set_A_v2 | proGenomes 2.1 | 0-5 | 1000 | 1000 | 13 | 0.013 | 0.000 | 0.920 |
| set_B_v2 | proGenomes 2.1 | 0-5 | 200 | 200 | 6 | 0.030 | 0.000 | 0.920 |
| set_B_v2 | proGenomes 2.1 | 5-10 | 2 | 2 | 2 | 1.000 | 0.990 | 0.770 |
| set_B_v2 | proGenomes 2.1 | 10-20 | 172 | 172 | 171 | 0.994 | 1.000 | 0.910 |
| set_B_v2 | proGenomes 2.1 | 20-40 | 202 | 202 | 202 | 1.000 | 1.000 | 0.900 |
| set_B_v2 | proGenomes 2.1 | 40-70 | 241 | 241 | 241 | 1.000 | 1.000 | 0.880 |
| set_B_v2 | proGenomes 2.1 | 70-100 | 183 | 183 | 183 | 1.000 | 1.000 | 0.880 |
| set_C_clean | proGenomes 2.1 | 0-5 | 52 | 52 | 9 | 0.173 | 0.105 | 0.425 |
| set_C_clean | proGenomes 2.1 | 5-10 | 48 | 48 | 22 | 0.458 | 0.395 | 0.435 |
| set_C_clean | proGenomes 2.1 | 10-20 | 101 | 101 | 81 | 0.802 | 0.750 | 0.470 |
| set_C_clean | proGenomes 2.1 | 20-40 | 244 | 244 | 207 | 0.848 | 0.750 | 0.510 |
| set_C_clean | proGenomes 2.1 | 40-70 | 354 | 354 | 343 | 0.969 | 0.850 | 0.580 |
| set_C_clean | proGenomes 2.1 | 70-100 | 201 | 201 | 200 | 0.995 | 0.880 | 0.620 |
| set_C_clean | GTDB r95 | 0-5 | 52 | 52 | 22 | 0.423 | 0.285 | 0.515 |
| set_C_clean | GTDB r95 | 5-10 | 48 | 48 | 47 | 0.979 | 1.000 | 0.535 |
| set_C_clean | GTDB r95 | 10-20 | 101 | 101 | 101 | 1.000 | 1.000 | 0.580 |
| set_C_clean | GTDB r95 | 20-40 | 244 | 244 | 244 | 1.000 | 1.000 | 0.610 |
| set_C_clean | GTDB r95 | 40-70 | 354 | 354 | 354 | 1.000 | 1.000 | 0.660 |
| set_C_clean | GTDB r95 | 70-100 | 201 | 201 | 201 | 1.000 | 1.000 | 0.680 |
| set_D_clean | proGenomes 2.1 | 0-5 | 62 | 62 | 39 | 0.629 | 0.900 | 0.825 |
| set_D_clean | proGenomes 2.1 | 5-10 | 51 | 51 | 48 | 0.941 | 1.000 | 0.890 |
| set_D_clean | proGenomes 2.1 | 10-20 | 104 | 104 | 101 | 0.971 | 1.000 | 0.790 |
| set_D_clean | proGenomes 2.1 | 20-40 | 195 | 195 | 195 | 1.000 | 1.000 | 0.850 |
| set_D_clean | proGenomes 2.1 | 40-70 | 420 | 420 | 420 | 1.000 | 1.000 | 0.840 |
| set_D_clean | proGenomes 2.1 | 70-100 | 168 | 168 | 168 | 1.000 | 1.000 | 0.820 |
| set_D_clean | GTDB r95 | 0-5 | 62 | 62 | 40 | 0.645 | 0.990 | 0.925 |
| set_D_clean | GTDB r95 | 5-10 | 51 | 51 | 51 | 1.000 | 1.000 | 0.940 |
| set_D_clean | GTDB r95 | 10-20 | 104 | 104 | 104 | 1.000 | 1.000 | 0.920 |
| set_D_clean | GTDB r95 | 20-40 | 195 | 195 | 195 | 1.000 | 1.000 | 0.920 |
| set_D_clean | GTDB r95 | 40-70 | 420 | 420 | 420 | 1.000 | 1.000 | 0.915 |
| set_D_clean | GTDB r95 | 70-100 | 168 | 168 | 168 | 1.000 | 1.000 | 0.900 |
| set_E | proGenomes 2.1 | 0-5 | 252 | 252 | 43 | 0.171 | 0.000 | 0.930 |
| set_E | proGenomes 2.1 | 5-10 | 29 | 29 | 28 | 0.966 | 1.000 | 0.910 |
| set_E | proGenomes 2.1 | 10-20 | 87 | 87 | 79 | 0.908 | 1.000 | 0.900 |
| set_E | proGenomes 2.1 | 20-40 | 153 | 153 | 149 | 0.974 | 1.000 | 0.890 |
| set_E | proGenomes 2.1 | 40-70 | 234 | 234 | 224 | 0.957 | 1.000 | 0.885 |
| set_E | proGenomes 2.1 | 70-100 | 245 | 245 | 238 | 0.971 | 1.000 | 0.880 |
| set_E | GTDB r95 | 0-5 | 252 | 252 | 44 | 0.175 | 0.000 | 0.950 |
| set_E | GTDB r95 | 5-10 | 29 | 29 | 29 | 1.000 | 1.000 | 0.940 |
| set_E | GTDB r95 | 10-20 | 87 | 87 | 83 | 0.954 | 1.000 | 0.940 |
| set_E | GTDB r95 | 20-40 | 153 | 153 | 150 | 0.980 | 1.000 | 0.930 |
| set_E | GTDB r95 | 40-70 | 234 | 234 | 226 | 0.966 | 1.000 | 0.920 |
| set_E | GTDB r95 | 70-100 | 245 | 245 | 240 | 0.980 | 1.000 | 0.920 |

### 2.2 stratum = `powered`

| Set | Database | band (%) | n | scored | fail | fail rate | median CSS | median RRS |
|---|---|---|---|---|---|---|---|---|
| set_A_v2 | proGenomes 2.1 | 0-5 | 984 | 984 | 13 | 0.013 | 0.000 | 0.920 |
| set_B_v2 | proGenomes 2.1 | 0-5 | 197 | 197 | 6 | 0.030 | 0.000 | 0.930 |
| set_B_v2 | proGenomes 2.1 | 5-10 | 2 | 2 | 2 | 1.000 | 0.990 | 0.770 |
| set_B_v2 | proGenomes 2.1 | 10-20 | 171 | 171 | 171 | 1.000 | 1.000 | 0.910 |
| set_B_v2 | proGenomes 2.1 | 20-40 | 200 | 200 | 200 | 1.000 | 1.000 | 0.900 |
| set_B_v2 | proGenomes 2.1 | 40-70 | 240 | 240 | 240 | 1.000 | 1.000 | 0.880 |
| set_B_v2 | proGenomes 2.1 | 70-100 | 183 | 183 | 183 | 1.000 | 1.000 | 0.880 |
| set_C_clean | proGenomes 2.1 | 0-5 | 15 | 15 | 4 | 0.267 | 0.120 | 0.620 |
| set_C_clean | proGenomes 2.1 | 5-10 | 17 | 17 | 13 | 0.765 | 1.000 | 0.850 |
| set_C_clean | proGenomes 2.1 | 10-20 | 47 | 47 | 45 | 0.957 | 1.000 | 0.890 |
| set_C_clean | proGenomes 2.1 | 20-40 | 138 | 138 | 129 | 0.935 | 0.840 | 0.590 |
| set_C_clean | proGenomes 2.1 | 40-70 | 283 | 283 | 279 | 0.986 | 0.870 | 0.620 |
| set_C_clean | proGenomes 2.1 | 70-100 | 181 | 181 | 180 | 0.995 | 0.890 | 0.630 |
| set_C_clean | GTDB r95 | 0-5 | 31 | 31 | 14 | 0.452 | 0.060 | 0.640 |
| set_C_clean | GTDB r95 | 5-10 | 34 | 34 | 33 | 0.971 | 1.000 | 0.585 |
| set_C_clean | GTDB r95 | 10-20 | 73 | 73 | 73 | 1.000 | 1.000 | 0.670 |
| set_C_clean | GTDB r95 | 20-40 | 220 | 220 | 220 | 1.000 | 1.000 | 0.630 |
| set_C_clean | GTDB r95 | 40-70 | 328 | 328 | 328 | 1.000 | 1.000 | 0.670 |
| set_C_clean | GTDB r95 | 70-100 | 194 | 194 | 194 | 1.000 | 1.000 | 0.690 |
| set_D_clean | proGenomes 2.1 | 0-5 | 57 | 57 | 39 | 0.684 | 0.960 | 0.860 |
| set_D_clean | proGenomes 2.1 | 5-10 | 47 | 47 | 47 | 1.000 | 1.000 | 0.900 |
| set_D_clean | proGenomes 2.1 | 10-20 | 93 | 93 | 93 | 1.000 | 1.000 | 0.800 |
| set_D_clean | proGenomes 2.1 | 20-40 | 184 | 184 | 184 | 1.000 | 1.000 | 0.860 |
| set_D_clean | proGenomes 2.1 | 40-70 | 405 | 405 | 405 | 1.000 | 1.000 | 0.850 |
| set_D_clean | proGenomes 2.1 | 70-100 | 159 | 159 | 159 | 1.000 | 1.000 | 0.830 |
| set_D_clean | GTDB r95 | 0-5 | 59 | 59 | 40 | 0.678 | 1.000 | 0.930 |
| set_D_clean | GTDB r95 | 5-10 | 49 | 49 | 49 | 1.000 | 1.000 | 0.950 |
| set_D_clean | GTDB r95 | 10-20 | 97 | 97 | 97 | 1.000 | 1.000 | 0.930 |
| set_D_clean | GTDB r95 | 20-40 | 190 | 190 | 190 | 1.000 | 1.000 | 0.930 |
| set_D_clean | GTDB r95 | 40-70 | 414 | 414 | 414 | 1.000 | 1.000 | 0.920 |
| set_D_clean | GTDB r95 | 70-100 | 165 | 165 | 165 | 1.000 | 1.000 | 0.900 |
| set_E | proGenomes 2.1 | 0-5 | 244 | 244 | 42 | 0.172 | 0.000 | 0.930 |
| set_E | proGenomes 2.1 | 5-10 | 29 | 29 | 28 | 0.966 | 1.000 | 0.910 |
| set_E | proGenomes 2.1 | 10-20 | 85 | 85 | 79 | 0.929 | 1.000 | 0.910 |
| set_E | proGenomes 2.1 | 20-40 | 153 | 153 | 149 | 0.974 | 1.000 | 0.890 |
| set_E | proGenomes 2.1 | 40-70 | 231 | 231 | 221 | 0.957 | 1.000 | 0.890 |
| set_E | proGenomes 2.1 | 70-100 | 244 | 244 | 237 | 0.971 | 1.000 | 0.880 |
| set_E | GTDB r95 | 0-5 | 246 | 246 | 43 | 0.175 | 0.000 | 0.950 |
| set_E | GTDB r95 | 5-10 | 29 | 29 | 29 | 1.000 | 1.000 | 0.940 |
| set_E | GTDB r95 | 10-20 | 87 | 87 | 83 | 0.954 | 1.000 | 0.940 |
| set_E | GTDB r95 | 20-40 | 153 | 153 | 150 | 0.980 | 1.000 | 0.930 |
| set_E | GTDB r95 | 40-70 | 232 | 232 | 224 | 0.966 | 1.000 | 0.920 |
| set_E | GTDB r95 | 70-100 | 245 | 245 | 240 | 0.980 | 1.000 | 0.920 |

### 2.3 stratum = `unpowered`

| Set | Database | band (%) | n | scored | fail | fail rate | median CSS | median RRS |
|---|---|---|---|---|---|---|---|---|
| set_A_v2 | proGenomes 2.1 | 0-5 | 16 | 16 | 0 | 0.000 | 0.015 | 0.390 |
| set_B_v2 | proGenomes 2.1 | 0-5 | 3 | 3 | 0 | 0.000 | 0.000 | 0.450 |
| set_B_v2 | proGenomes 2.1 | 10-20 | 1 | 1 | 0 | 0.000 | 0.440 | 0.490 |
| set_B_v2 | proGenomes 2.1 | 20-40 | 2 | 2 | 2 | 1.000 | 0.610 | 0.355 |
| set_B_v2 | proGenomes 2.1 | 40-70 | 1 | 1 | 1 | 1.000 | 1.000 | 0.470 |
| set_C_clean | proGenomes 2.1 | 0-5 | 37 | 37 | 5 | 0.135 | 0.100 | 0.380 |
| set_C_clean | proGenomes 2.1 | 5-10 | 31 | 31 | 9 | 0.290 | 0.350 | 0.380 |
| set_C_clean | proGenomes 2.1 | 10-20 | 54 | 54 | 36 | 0.667 | 0.560 | 0.400 |
| set_C_clean | proGenomes 2.1 | 20-40 | 106 | 106 | 78 | 0.736 | 0.625 | 0.430 |
| set_C_clean | proGenomes 2.1 | 40-70 | 71 | 71 | 64 | 0.901 | 0.680 | 0.450 |
| set_C_clean | proGenomes 2.1 | 70-100 | 20 | 20 | 20 | 1.000 | 0.780 | 0.445 |
| set_C_clean | GTDB r95 | 0-5 | 21 | 21 | 8 | 0.381 | 0.310 | 0.450 |
| set_C_clean | GTDB r95 | 5-10 | 14 | 14 | 14 | 1.000 | 0.820 | 0.460 |
| set_C_clean | GTDB r95 | 10-20 | 28 | 28 | 28 | 1.000 | 0.960 | 0.445 |
| set_C_clean | GTDB r95 | 20-40 | 24 | 24 | 24 | 1.000 | 0.995 | 0.455 |
| set_C_clean | GTDB r95 | 40-70 | 26 | 26 | 26 | 1.000 | 0.975 | 0.440 |
| set_C_clean | GTDB r95 | 70-100 | 7 | 7 | 7 | 1.000 | 0.970 | 0.460 |
| set_D_clean | proGenomes 2.1 | 0-5 | 5 | 5 | 0 | 0.000 | 0.110 | 0.440 |
| set_D_clean | proGenomes 2.1 | 5-10 | 4 | 4 | 1 | 0.250 | 0.270 | 0.235 |
| set_D_clean | proGenomes 2.1 | 10-20 | 11 | 11 | 8 | 0.727 | 0.560 | 0.310 |
| set_D_clean | proGenomes 2.1 | 20-40 | 11 | 11 | 11 | 1.000 | 0.690 | 0.410 |
| set_D_clean | proGenomes 2.1 | 40-70 | 15 | 15 | 15 | 1.000 | 0.780 | 0.430 |
| set_D_clean | proGenomes 2.1 | 70-100 | 9 | 9 | 9 | 1.000 | 0.870 | 0.450 |
| set_D_clean | GTDB r95 | 0-5 | 3 | 3 | 0 | 0.000 | 0.190 | 0.460 |
| set_D_clean | GTDB r95 | 5-10 | 2 | 2 | 2 | 1.000 | 0.625 | 0.300 |
| set_D_clean | GTDB r95 | 10-20 | 7 | 7 | 7 | 1.000 | 0.690 | 0.360 |
| set_D_clean | GTDB r95 | 20-40 | 5 | 5 | 5 | 1.000 | 0.880 | 0.450 |
| set_D_clean | GTDB r95 | 40-70 | 6 | 6 | 6 | 1.000 | 0.985 | 0.440 |
| set_D_clean | GTDB r95 | 70-100 | 3 | 3 | 3 | 1.000 | 1.000 | 0.400 |
| set_E | proGenomes 2.1 | 0-5 | 8 | 8 | 1 | 0.125 | 0.050 | 0.355 |
| set_E | proGenomes 2.1 | 10-20 | 2 | 2 | 0 | 0.000 | 0.390 | 0.440 |
| set_E | proGenomes 2.1 | 40-70 | 3 | 3 | 3 | 1.000 | 0.970 | 0.330 |
| set_E | proGenomes 2.1 | 70-100 | 1 | 1 | 1 | 1.000 | 0.970 | 0.470 |
| set_E | GTDB r95 | 0-5 | 6 | 6 | 1 | 0.167 | 0.045 | 0.400 |
| set_E | GTDB r95 | 40-70 | 2 | 2 | 2 | 1.000 | 0.745 | 0.425 |

## 3. Detection agreement at the MIMAG-inspired 5 % and 10 % contamination thresholds

95 % CIs: cluster bootstrap over the reference genomes (`dominant_accession`), 2000 resamples, seed 7600 — the 10 simulations per reference are not independent.


### 3.1 stratum = `all`

| Set | Database | thr | detector | TP | FP | TN | FN | sensitivity/recall (95 % CI) | specificity (95 % CI) | precision (95 % CI) | F1 | balanced acc. | MCC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| set_B_v2 | proGenomes 2.1 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 799 | 6 | 194 | 1 | 0.999 (0.996–1.000) | 0.970 (0.944–0.990) | 0.993 (0.986–0.998) | 0.996 | 0.984 | 0.978 |
| set_B_v2 | proGenomes 2.1 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 800 | 6 | 194 | 0 | 1.000 (1.000–1.000) | 0.970 (0.945–0.991) | 0.993 (0.986–0.998) | 0.996 | 0.985 | 0.981 |
| set_B_v2 | proGenomes 2.1 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 741 | 0 | 200 | 59 | 0.926 (0.908–0.944) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.962 | 0.963 | 0.846 |
| set_B_v2 | proGenomes 2.1 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 797 | 8 | 194 | 1 | 0.999 (0.996–1.000) | 0.960 (0.932–0.985) | 0.990 (0.983–0.996) | 0.994 | 0.980 | 0.972 |
| set_B_v2 | proGenomes 2.1 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 797 | 4 | 198 | 1 | 0.999 (0.996–1.000) | 0.980 (0.958–0.995) | 0.995 (0.990–0.999) | 0.997 | 0.990 | 0.984 |
| set_B_v2 | proGenomes 2.1 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 651 | 0 | 202 | 147 | 0.816 (0.788–0.845) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.899 | 0.908 | 0.687 |
| set_C_clean | proGenomes 2.1 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 853 | 9 | 43 | 95 | 0.900 (0.864–0.933) | 0.827 (0.706–0.922) | 0.990 (0.983–0.995) | 0.943 | 0.863 | 0.468 |
| set_C_clean | proGenomes 2.1 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 929 | 23 | 29 | 19 | 0.980 (0.972–0.988) | 0.558 (0.390–0.727) | 0.976 (0.963–0.987) | 0.978 | 0.769 | 0.558 |
| set_C_clean | proGenomes 2.1 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 455 | 0 | 52 | 493 | 0.480 (0.445–0.517) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.649 | 0.740 | 0.214 |
| set_C_clean | proGenomes 2.1 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 831 | 31 | 69 | 69 | 0.923 (0.889–0.954) | 0.690 (0.579–0.786) | 0.964 (0.951–0.976) | 0.943 | 0.807 | 0.533 |
| set_C_clean | proGenomes 2.1 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 879 | 35 | 65 | 21 | 0.977 (0.965–0.987) | 0.650 (0.544–0.752) | 0.962 (0.947–0.975) | 0.969 | 0.813 | 0.671 |
| set_C_clean | proGenomes 2.1 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 161 | 0 | 100 | 739 | 0.179 (0.150–0.210) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.303 | 0.589 | 0.146 |
| set_C_clean | GTDB r95 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 947 | 22 | 30 | 1 | 0.999 (0.997–1.000) | 0.577 (0.453–0.710) | 0.977 (0.968–0.987) | 0.988 | 0.788 | 0.738 |
| set_C_clean | GTDB r95 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 929 | 23 | 29 | 19 | 0.980 (0.972–0.987) | 0.558 (0.389–0.729) | 0.976 (0.964–0.987) | 0.978 | 0.769 | 0.558 |
| set_C_clean | GTDB r95 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 455 | 0 | 52 | 493 | 0.480 (0.443–0.514) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.649 | 0.740 | 0.214 |
| set_C_clean | GTDB r95 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 900 | 69 | 31 | 0 | 1.000 (1.000–1.000) | 0.310 (0.228–0.395) | 0.929 (0.915–0.942) | 0.963 | 0.655 | 0.537 |
| set_C_clean | GTDB r95 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 879 | 35 | 65 | 21 | 0.977 (0.965–0.987) | 0.650 (0.551–0.753) | 0.962 (0.947–0.975) | 0.969 | 0.813 | 0.671 |
| set_C_clean | GTDB r95 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 161 | 0 | 100 | 739 | 0.179 (0.150–0.211) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.303 | 0.589 | 0.146 |
| set_D_clean | proGenomes 2.1 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 932 | 39 | 23 | 6 | 0.994 (0.986–0.999) | 0.371 (0.257–0.500) | 0.960 (0.947–0.971) | 0.976 | 0.682 | 0.524 |
| set_D_clean | proGenomes 2.1 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 927 | 9 | 53 | 11 | 0.988 (0.981–0.995) | 0.855 (0.754–0.945) | 0.990 (0.983–0.997) | 0.989 | 0.922 | 0.831 |
| set_D_clean | proGenomes 2.1 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 642 | 0 | 62 | 296 | 0.684 (0.642–0.726) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.813 | 0.842 | 0.344 |
| set_D_clean | proGenomes 2.1 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 884 | 87 | 26 | 3 | 0.997 (0.992–1.000) | 0.230 (0.154–0.315) | 0.910 (0.892–0.929) | 0.952 | 0.613 | 0.428 |
| set_D_clean | proGenomes 2.1 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 866 | 12 | 101 | 21 | 0.976 (0.967–0.985) | 0.894 (0.826–0.948) | 0.986 (0.977–0.993) | 0.981 | 0.935 | 0.842 |
| set_D_clean | proGenomes 2.1 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 380 | 0 | 113 | 507 | 0.428 (0.380–0.474) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.600 | 0.714 | 0.279 |
| set_D_clean | GTDB r95 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 938 | 40 | 22 | 0 | 1.000 (1.000–1.000) | 0.355 (0.236–0.483) | 0.959 (0.946–0.971) | 0.979 | 0.677 | 0.583 |
| set_D_clean | GTDB r95 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 927 | 9 | 53 | 11 | 0.988 (0.982–0.995) | 0.855 (0.754–0.943) | 0.990 (0.983–0.997) | 0.989 | 0.922 | 0.831 |
| set_D_clean | GTDB r95 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 642 | 0 | 62 | 296 | 0.684 (0.641–0.726) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.813 | 0.842 | 0.344 |
| set_D_clean | GTDB r95 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 887 | 91 | 22 | 0 | 1.000 (1.000–1.000) | 0.195 (0.121–0.276) | 0.907 (0.888–0.925) | 0.951 | 0.597 | 0.420 |
| set_D_clean | GTDB r95 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 866 | 12 | 101 | 21 | 0.976 (0.967–0.985) | 0.894 (0.826–0.955) | 0.986 (0.977–0.994) | 0.981 | 0.935 | 0.842 |
| set_D_clean | GTDB r95 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 380 | 0 | 113 | 507 | 0.428 (0.382–0.474) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.600 | 0.714 | 0.279 |
| set_E | proGenomes 2.1 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 718 | 43 | 209 | 30 | 0.960 (0.945–0.974) | 0.829 (0.778–0.878) | 0.944 (0.925–0.960) | 0.952 | 0.895 | 0.803 |
| set_E | proGenomes 2.1 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 734 | 9 | 243 | 14 | 0.981 (0.971–0.990) | 0.964 (0.941–0.985) | 0.988 (0.980–0.995) | 0.985 | 0.973 | 0.940 |
| set_E | proGenomes 2.1 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 691 | 5 | 247 | 57 | 0.924 (0.905–0.943) | 0.980 (0.961–0.996) | 0.993 (0.986–0.999) | 0.957 | 0.952 | 0.853 |
| set_E | proGenomes 2.1 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 690 | 71 | 210 | 29 | 0.960 (0.944–0.973) | 0.747 (0.696–0.797) | 0.907 (0.885–0.927) | 0.932 | 0.854 | 0.745 |
| set_E | proGenomes 2.1 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 700 | 9 | 272 | 19 | 0.974 (0.961–0.984) | 0.968 (0.945–0.986) | 0.987 (0.979–0.994) | 0.980 | 0.971 | 0.932 |
| set_E | proGenomes 2.1 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 609 | 1 | 280 | 110 | 0.847 (0.818–0.875) | 0.996 (0.989–1.000) | 0.998 (0.995–1.000) | 0.916 | 0.922 | 0.777 |
| set_E | GTDB r95 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 728 | 44 | 208 | 20 | 0.973 (0.961–0.984) | 0.825 (0.779–0.872) | 0.943 (0.925–0.959) | 0.958 | 0.899 | 0.827 |
| set_E | GTDB r95 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 734 | 9 | 243 | 14 | 0.981 (0.972–0.991) | 0.964 (0.940–0.985) | 0.988 (0.980–0.995) | 0.985 | 0.973 | 0.940 |
| set_E | GTDB r95 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 691 | 5 | 247 | 57 | 0.924 (0.903–0.942) | 0.980 (0.961–0.996) | 0.993 (0.986–0.999) | 0.957 | 0.952 | 0.853 |
| set_E | GTDB r95 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 699 | 73 | 208 | 20 | 0.972 (0.959–0.983) | 0.740 (0.689–0.791) | 0.905 (0.885–0.925) | 0.938 | 0.856 | 0.763 |
| set_E | GTDB r95 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 700 | 9 | 272 | 19 | 0.974 (0.962–0.984) | 0.968 (0.946–0.987) | 0.987 (0.978–0.995) | 0.980 | 0.971 | 0.932 |
| set_E | GTDB r95 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 609 | 1 | 280 | 110 | 0.847 (0.819–0.874) | 0.996 (0.989–1.000) | 0.998 (0.995–1.000) | 0.916 | 0.922 | 0.777 |

### 3.2 stratum = `powered`

| Set | Database | thr | detector | TP | FP | TN | FN | sensitivity/recall (95 % CI) | specificity (95 % CI) | precision (95 % CI) | F1 | balanced acc. | MCC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| set_B_v2 | proGenomes 2.1 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 796 | 6 | 191 | 0 | 1.000 (1.000–1.000) | 0.970 (0.942–0.990) | 0.993 (0.986–0.998) | 0.996 | 0.985 | 0.981 |
| set_B_v2 | proGenomes 2.1 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 796 | 5 | 192 | 0 | 1.000 (1.000–1.000) | 0.975 (0.949–0.995) | 0.994 (0.988–0.999) | 0.997 | 0.987 | 0.984 |
| set_B_v2 | proGenomes 2.1 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 737 | 0 | 197 | 59 | 0.926 (0.907–0.943) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.962 | 0.963 | 0.844 |
| set_B_v2 | proGenomes 2.1 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 794 | 8 | 191 | 0 | 1.000 (1.000–1.000) | 0.960 (0.929–0.985) | 0.990 (0.982–0.996) | 0.995 | 0.980 | 0.975 |
| set_B_v2 | proGenomes 2.1 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 793 | 3 | 196 | 1 | 0.999 (0.996–1.000) | 0.985 (0.967–1.000) | 0.996 (0.991–1.000) | 0.998 | 0.992 | 0.987 |
| set_B_v2 | proGenomes 2.1 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 649 | 0 | 199 | 145 | 0.817 (0.789–0.846) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.899 | 0.909 | 0.688 |
| set_C_clean | proGenomes 2.1 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 646 | 4 | 11 | 20 | 0.970 (0.954–0.985) | 0.733 (0.467–0.941) | 0.994 (0.988–0.999) | 0.982 | 0.852 | 0.495 |
| set_C_clean | proGenomes 2.1 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 662 | 6 | 9 | 4 | 0.994 (0.988–0.999) | 0.600 (0.347–0.846) | 0.991 (0.984–0.997) | 0.993 | 0.797 | 0.637 |
| set_C_clean | proGenomes 2.1 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 346 | 0 | 15 | 320 | 0.519 (0.479–0.564) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.684 | 0.760 | 0.152 |
| set_C_clean | proGenomes 2.1 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 633 | 17 | 15 | 16 | 0.975 (0.959–0.989) | 0.469 (0.273–0.667) | 0.974 (0.960–0.986) | 0.975 | 0.722 | 0.451 |
| set_C_clean | proGenomes 2.1 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 642 | 12 | 20 | 7 | 0.989 (0.979–0.997) | 0.625 (0.464–0.800) | 0.982 (0.970–0.992) | 0.985 | 0.807 | 0.666 |
| set_C_clean | proGenomes 2.1 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 114 | 0 | 32 | 535 | 0.176 (0.144–0.209) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.299 | 0.588 | 0.100 |
| set_C_clean | GTDB r95 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 848 | 14 | 17 | 1 | 0.999 (0.996–1.000) | 0.548 (0.379–0.706) | 0.984 (0.975–0.992) | 0.991 | 0.774 | 0.713 |
| set_C_clean | GTDB r95 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 837 | 13 | 18 | 12 | 0.986 (0.978–0.993) | 0.581 (0.364–0.778) | 0.985 (0.974–0.993) | 0.985 | 0.783 | 0.576 |
| set_C_clean | GTDB r95 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 417 | 0 | 31 | 432 | 0.491 (0.451–0.531) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.659 | 0.746 | 0.181 |
| set_C_clean | GTDB r95 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 815 | 47 | 18 | 0 | 1.000 (1.000–1.000) | 0.277 (0.185–0.364) | 0.946 (0.931–0.959) | 0.972 | 0.638 | 0.512 |
| set_C_clean | GTDB r95 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 802 | 23 | 42 | 13 | 0.984 (0.973–0.994) | 0.646 (0.516–0.782) | 0.972 (0.959–0.984) | 0.978 | 0.815 | 0.681 |
| set_C_clean | GTDB r95 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 146 | 0 | 65 | 669 | 0.179 (0.150–0.211) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.304 | 0.590 | 0.126 |
| set_D_clean | proGenomes 2.1 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 888 | 39 | 18 | 0 | 1.000 (1.000–1.000) | 0.316 (0.196–0.429) | 0.958 (0.946–0.970) | 0.979 | 0.658 | 0.550 |
| set_D_clean | proGenomes 2.1 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 878 | 8 | 49 | 10 | 0.989 (0.982–0.995) | 0.860 (0.756–0.948) | 0.991 (0.984–0.997) | 0.990 | 0.924 | 0.835 |
| set_D_clean | proGenomes 2.1 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 623 | 0 | 57 | 265 | 0.702 (0.661–0.741) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.825 | 0.851 | 0.352 |
| set_D_clean | proGenomes 2.1 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 841 | 86 | 18 | 0 | 1.000 (1.000–1.000) | 0.173 (0.105–0.248) | 0.907 (0.887–0.925) | 0.951 | 0.587 | 0.396 |
| set_D_clean | proGenomes 2.1 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 824 | 9 | 95 | 17 | 0.980 (0.970–0.988) | 0.913 (0.850–0.966) | 0.989 (0.981–0.996) | 0.985 | 0.947 | 0.865 |
| set_D_clean | proGenomes 2.1 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 372 | 0 | 104 | 469 | 0.442 (0.395–0.489) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.613 | 0.721 | 0.283 |
| set_D_clean | GTDB r95 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 915 | 40 | 19 | 0 | 1.000 (1.000–1.000) | 0.322 (0.205–0.451) | 0.958 (0.945–0.969) | 0.979 | 0.661 | 0.555 |
| set_D_clean | GTDB r95 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 905 | 9 | 50 | 10 | 0.989 (0.983–0.995) | 0.848 (0.733–0.942) | 0.990 (0.983–0.997) | 0.990 | 0.918 | 0.830 |
| set_D_clean | GTDB r95 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 634 | 0 | 59 | 281 | 0.693 (0.651–0.734) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.819 | 0.846 | 0.347 |
| set_D_clean | GTDB r95 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 866 | 89 | 19 | 0 | 1.000 (1.000–1.000) | 0.176 (0.105–0.259) | 0.907 (0.887–0.925) | 0.951 | 0.588 | 0.399 |
| set_D_clean | GTDB r95 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 848 | 12 | 96 | 18 | 0.979 (0.971–0.987) | 0.889 (0.821–0.953) | 0.986 (0.977–0.994) | 0.983 | 0.934 | 0.848 |
| set_D_clean | GTDB r95 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 377 | 0 | 108 | 489 | 0.435 (0.389–0.480) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.607 | 0.718 | 0.281 |
| set_E | proGenomes 2.1 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 714 | 42 | 202 | 28 | 0.962 (0.948–0.976) | 0.828 (0.779–0.875) | 0.944 (0.927–0.961) | 0.953 | 0.895 | 0.806 |
| set_E | proGenomes 2.1 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 728 | 6 | 238 | 14 | 0.981 (0.971–0.991) | 0.975 (0.955–0.992) | 0.992 (0.985–0.997) | 0.986 | 0.978 | 0.946 |
| set_E | proGenomes 2.1 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 685 | 5 | 239 | 57 | 0.923 (0.904–0.942) | 0.980 (0.961–0.996) | 0.993 (0.986–0.999) | 0.957 | 0.951 | 0.850 |
| set_E | proGenomes 2.1 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 686 | 70 | 203 | 27 | 0.962 (0.947–0.976) | 0.744 (0.692–0.795) | 0.907 (0.886–0.928) | 0.934 | 0.853 | 0.747 |
| set_E | proGenomes 2.1 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 694 | 7 | 266 | 19 | 0.973 (0.961–0.984) | 0.974 (0.954–0.992) | 0.990 (0.982–0.997) | 0.982 | 0.974 | 0.935 |
| set_E | proGenomes 2.1 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 605 | 1 | 272 | 108 | 0.849 (0.820–0.876) | 0.996 (0.988–1.000) | 0.998 (0.995–1.000) | 0.917 | 0.922 | 0.777 |
| set_E | GTDB r95 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 726 | 43 | 203 | 20 | 0.973 (0.961–0.984) | 0.825 (0.777–0.872) | 0.944 (0.927–0.960) | 0.958 | 0.899 | 0.826 |
| set_E | GTDB r95 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 732 | 7 | 239 | 14 | 0.981 (0.971–0.991) | 0.972 (0.950–0.991) | 0.991 (0.984–0.997) | 0.986 | 0.976 | 0.944 |
| set_E | GTDB r95 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 689 | 5 | 241 | 57 | 0.924 (0.903–0.942) | 0.980 (0.960–0.996) | 0.993 (0.986–0.999) | 0.957 | 0.952 | 0.851 |
| set_E | GTDB r95 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 697 | 72 | 203 | 20 | 0.972 (0.959–0.983) | 0.738 (0.684–0.790) | 0.906 (0.885–0.927) | 0.938 | 0.855 | 0.762 |
| set_E | GTDB r95 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 698 | 7 | 268 | 19 | 0.974 (0.961–0.984) | 0.975 (0.954–0.993) | 0.990 (0.982–0.997) | 0.982 | 0.974 | 0.936 |
| set_E | GTDB r95 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 607 | 1 | 274 | 110 | 0.847 (0.819–0.874) | 0.996 (0.989–1.000) | 0.998 (0.995–1.000) | 0.916 | 0.921 | 0.775 |

### 3.3 stratum = `unpowered`

| Set | Database | thr | detector | TP | FP | TN | FN | sensitivity/recall (95 % CI) | specificity (95 % CI) | precision (95 % CI) | F1 | balanced acc. | MCC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| set_C_clean | proGenomes 2.1 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 207 | 5 | 32 | 75 | 0.734 (0.640–0.816) | 0.865 (0.739–0.967) | 0.976 (0.958–0.995) | 0.838 | 0.799 | 0.406 |
| set_C_clean | proGenomes 2.1 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 267 | 17 | 20 | 15 | 0.947 (0.920–0.970) | 0.540 (0.324–0.759) | 0.940 (0.904–0.973) | 0.944 | 0.744 | 0.499 |
| set_C_clean | proGenomes 2.1 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 109 | 0 | 37 | 173 | 0.387 (0.326–0.451) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.557 | 0.693 | 0.261 |
| set_C_clean | proGenomes 2.1 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 198 | 14 | 54 | 53 | 0.789 (0.698–0.869) | 0.794 (0.682–0.894) | 0.934 (0.902–0.964) | 0.855 | 0.791 | 0.506 |
| set_C_clean | proGenomes 2.1 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 237 | 23 | 45 | 14 | 0.944 (0.910–0.973) | 0.662 (0.537–0.787) | 0.911 (0.875–0.946) | 0.928 | 0.803 | 0.639 |
| set_C_clean | proGenomes 2.1 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 47 | 0 | 68 | 204 | 0.187 (0.136–0.242) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.315 | 0.594 | 0.216 |
| set_C_clean | GTDB r95 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 99 | 8 | 13 | 0 | 1.000 (1.000–1.000) | 0.619 (0.400–0.809) | 0.925 (0.877–0.971) | 0.961 | 0.809 | 0.757 |
| set_C_clean | GTDB r95 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 92 | 10 | 11 | 7 | 0.929 (0.871–0.973) | 0.524 (0.227–0.826) | 0.902 (0.833–0.968) | 0.915 | 0.727 | 0.482 |
| set_C_clean | GTDB r95 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 38 | 0 | 21 | 61 | 0.384 (0.303–0.467) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.555 | 0.692 | 0.314 |
| set_C_clean | GTDB r95 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 85 | 22 | 13 | 0 | 1.000 (1.000–1.000) | 0.371 (0.200–0.531) | 0.794 (0.725–0.860) | 0.885 | 0.686 | 0.543 |
| set_C_clean | GTDB r95 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 77 | 12 | 23 | 8 | 0.906 (0.843–0.960) | 0.657 (0.500–0.811) | 0.865 (0.798–0.929) | 0.885 | 0.781 | 0.585 |
| set_C_clean | GTDB r95 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 15 | 0 | 35 | 70 | 0.176 (0.100–0.253) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.300 | 0.588 | 0.242 |
| set_D_clean | proGenomes 2.1 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 44 | 0 | 5 | 6 | 0.880 (0.818–0.966) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.936 | 0.940 | 0.632 |
| set_D_clean | proGenomes 2.1 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 49 | 1 | 4 | 1 | 0.980 (0.946–1.000) | 0.800 (0.333–1.000) | 0.980 (0.925–1.000) | 0.980 | 0.890 | 0.780 |
| set_D_clean | proGenomes 2.1 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 19 | 0 | 5 | 31 | 0.380 (0.197–0.571) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.551 | 0.690 | 0.230 |
| set_D_clean | proGenomes 2.1 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 43 | 1 | 8 | 3 | 0.935 (0.895–1.000) | 0.889 (0.700–1.000) | 0.977 (0.930–1.000) | 0.956 | 0.912 | 0.762 |
| set_D_clean | proGenomes 2.1 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 42 | 3 | 6 | 4 | 0.913 (0.877–0.968) | 0.667 (0.333–1.000) | 0.933 (0.829–1.000) | 0.923 | 0.790 | 0.556 |
| set_D_clean | proGenomes 2.1 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 8 | 0 | 9 | 38 | 0.174 (0.026–0.365) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.296 | 0.587 | 0.182 |
| set_D_clean | GTDB r95 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 23 | 0 | 3 | 0 | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 1.000 | 1.000 | 1.000 |
| set_D_clean | GTDB r95 | ≥5 % | MAGICC V5 (pred contamination >= 5 %) | 22 | 0 | 3 | 1 | 0.957 (0.842–1.000) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.978 | 0.978 | 0.847 |
| set_D_clean | GTDB r95 | ≥5 % | CheckM2 (pred contamination >= 5 %) | 8 | 0 | 3 | 15 | 0.348 (0.178–0.727) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.516 | 0.674 | 0.241 |
| set_D_clean | GTDB r95 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 21 | 2 | 3 | 0 | 1.000 (1.000–1.000) | 0.600 (0.167–1.000) | 0.913 (0.842–1.000) | 0.955 | 0.800 | 0.740 |
| set_D_clean | GTDB r95 | ≥10 % | MAGICC V5 (pred contamination >= 10 %) | 18 | 0 | 5 | 3 | 0.857 (0.733–0.952) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.923 | 0.929 | 0.732 |
| set_D_clean | GTDB r95 | ≥10 % | CheckM2 (pred contamination >= 10 %) | 3 | 0 | 5 | 18 | 0.143 (0.029–0.429) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.250 | 0.571 | 0.176 |

## 4. Threshold-free ranking: Spearman CSS vs true contamination, beside MAGICC V5 and CheckM2

This is the section that puts the three tools on equal footing: no threshold is applied to any of them. `p_cluster_level` uses the number of reference-genome clusters as the effective n (the conservative choice) and is Benjamini–Hochberg corrected across every row of this table.

**Two reading instructions, both necessary to avoid misinterpreting this table.** (i) A low Spearman ρ for CSS is *not* a failure of detection: CSS **saturates at 1.0** once a genome is confidently chimeric, so it carries almost no rank information *within* the contaminated range even where it separates contaminated from clean genomes well — compare ρ with the AUROC columns, which are the detection-relevant numbers. This is precisely why GUNC must be reported as a detector and never placed in an MAE table. (ii) CSS ρ can be *higher* in the `unpowered` stratum than in the `powered` one for the same reason inverted: attenuated (or forcibly zeroed) scores are un-saturated and therefore more rank-informative, while being *less* reliable as calls. Do not read a higher unpowered ρ as better performance.

GUNC also emits `contamination_portion`, a continuous quantity that is a far better rank-predictor of true contamination than CSS. It is reported here for completeness, but it is **not** GUNC's published output for quality assessment and is not calibrated as a contamination percentage, so it does not enter the MAE table either.


### 4.1 stratum = `all`

| Set | Database | score | n | clusters | Spearman ρ (95 % CI) | BH q | AUROC ≥5 % | AUROC ≥10 % |
|---|---|---|---|---|---|---|---|---|
| set_B_v2 | proGenomes 2.1 | GUNC 1.1.1 CSS | 1000 | 803 | 0.572 (0.517–0.620) | 0.000 | 0.999 (0.999–1.000) | 0.996 (0.989–1.000) |
| set_B_v2 | proGenomes 2.1 | GUNC 1.1.1 contamination_portion | 1000 | 803 | 0.925 (0.906–0.941) | 0.000 | 0.984 (0.967–0.996) | 0.984 (0.969–0.996) |
| set_B_v2 | proGenomes 2.1 | MAGICC V5 predicted contamination | 1000 | 803 | 0.959 (0.952–0.963) | 0.000 | 0.997 (0.993–1.000) | 0.997 (0.993–1.000) |
| set_B_v2 | proGenomes 2.1 | CheckM2 predicted contamination | 1000 | 803 | 0.794 (0.761–0.822) | 0.000 | 0.996 (0.993–0.999) | 0.996 (0.992–0.999) |
| set_C_clean | proGenomes 2.1 | GUNC 1.1.1 CSS | 1000 | 100 | 0.321 (0.238–0.415) | 0.001 | 0.889 (0.811–0.957) | 0.814 (0.755–0.870) |
| set_C_clean | proGenomes 2.1 | GUNC 1.1.1 contamination_portion | 1000 | 100 | 0.366 (0.247–0.484) | 0.000 | 0.710 (0.594–0.834) | 0.725 (0.646–0.803) |
| set_C_clean | proGenomes 2.1 | MAGICC V5 predicted contamination | 1000 | 100 | 0.891 (0.860–0.919) | 0.000 | 0.957 (0.938–0.976) | 0.965 (0.950–0.979) |
| set_C_clean | proGenomes 2.1 | CheckM2 predicted contamination | 1000 | 100 | 0.610 (0.566–0.654) | 0.000 | 0.908 (0.878–0.936) | 0.898 (0.873–0.920) |
| set_C_clean | GTDB r95 | GUNC 1.1.1 CSS | 1000 | 100 | 0.200 (0.134–0.269) | 0.052 | 0.832 (0.755–0.904) | 0.711 (0.650–0.771) |
| set_C_clean | GTDB r95 | GUNC 1.1.1 contamination_portion | 1000 | 100 | 0.847 (0.817–0.872) | 0.000 | 0.988 (0.980–0.994) | 0.986 (0.978–0.992) |
| set_C_clean | GTDB r95 | MAGICC V5 predicted contamination | 1000 | 100 | 0.891 (0.858–0.919) | 0.000 | 0.957 (0.937–0.977) | 0.965 (0.951–0.978) |
| set_C_clean | GTDB r95 | CheckM2 predicted contamination | 1000 | 100 | 0.610 (0.562–0.653) | 0.000 | 0.908 (0.878–0.935) | 0.898 (0.873–0.921) |
| set_D_clean | proGenomes 2.1 | GUNC 1.1.1 CSS | 1000 | 100 | 0.103 (0.041–0.168) | 0.327 | 0.703 (0.630–0.777) | 0.593 (0.538–0.649) |
| set_D_clean | proGenomes 2.1 | GUNC 1.1.1 contamination_portion | 1000 | 100 | 0.788 (0.716–0.848) | 0.000 | 0.960 (0.921–0.986) | 0.959 (0.924–0.985) |
| set_D_clean | proGenomes 2.1 | MAGICC V5 predicted contamination | 1000 | 100 | 0.933 (0.917–0.947) | 0.000 | 0.988 (0.975–0.997) | 0.987 (0.976–0.995) |
| set_D_clean | proGenomes 2.1 | CheckM2 predicted contamination | 1000 | 100 | 0.693 (0.648–0.733) | 0.000 | 0.940 (0.919–0.958) | 0.927 (0.908–0.946) |
| set_D_clean | GTDB r95 | GUNC 1.1.1 CSS | 1000 | 100 | 0.199 (0.135–0.262) | 0.053 | 0.726 (0.660–0.791) | 0.633 (0.589–0.683) |
| set_D_clean | GTDB r95 | GUNC 1.1.1 contamination_portion | 1000 | 100 | 0.858 (0.825–0.885) | 0.000 | 0.993 (0.986–0.998) | 0.993 (0.988–0.997) |
| set_D_clean | GTDB r95 | MAGICC V5 predicted contamination | 1000 | 100 | 0.933 (0.917–0.947) | 0.000 | 0.988 (0.974–0.997) | 0.987 (0.975–0.995) |
| set_D_clean | GTDB r95 | CheckM2 predicted contamination | 1000 | 100 | 0.693 (0.653–0.733) | 0.000 | 0.940 (0.918–0.958) | 0.927 (0.907–0.947) |
| set_E | proGenomes 2.1 | GUNC 1.1.1 CSS | 1000 | 785 | 0.559 (0.506–0.608) | 0.000 | 0.913 (0.886–0.939) | 0.869 (0.839–0.897) |
| set_E | proGenomes 2.1 | GUNC 1.1.1 contamination_portion | 1000 | 785 | 0.837 (0.805–0.867) | 0.000 | 0.950 (0.932–0.966) | 0.951 (0.934–0.966) |
| set_E | proGenomes 2.1 | MAGICC V5 predicted contamination | 1000 | 785 | 0.960 (0.951–0.966) | 0.000 | 0.994 (0.990–0.998) | 0.996 (0.993–0.998) |
| set_E | proGenomes 2.1 | CheckM2 predicted contamination | 1000 | 785 | 0.846 (0.822–0.869) | 0.000 | 0.992 (0.988–0.996) | 0.988 (0.982–0.992) |
| set_E | GTDB r95 | GUNC 1.1.1 CSS | 1000 | 785 | 0.622 (0.575–0.668) | 0.000 | 0.923 (0.899–0.946) | 0.879 (0.852–0.904) |
| set_E | GTDB r95 | GUNC 1.1.1 contamination_portion | 1000 | 785 | 0.863 (0.835–0.888) | 0.000 | 0.971 (0.958–0.981) | 0.969 (0.956–0.979) |
| set_E | GTDB r95 | MAGICC V5 predicted contamination | 1000 | 785 | 0.960 (0.951–0.967) | 0.000 | 0.994 (0.990–0.998) | 0.996 (0.993–0.998) |
| set_E | GTDB r95 | CheckM2 predicted contamination | 1000 | 785 | 0.846 (0.822–0.868) | 0.000 | 0.992 (0.988–0.996) | 0.988 (0.982–0.993) |

### 4.2 stratum = `powered`

| Set | Database | score | n | clusters | Spearman ρ (95 % CI) | BH q | AUROC ≥5 % | AUROC ≥10 % |
|---|---|---|---|---|---|---|---|---|
| set_B_v2 | proGenomes 2.1 | GUNC 1.1.1 CSS | 993 | 798 | 0.567 (0.514–0.620) | 0.000 | 1.000 (0.999–1.000) | 0.996 (0.989–1.000) |
| set_B_v2 | proGenomes 2.1 | GUNC 1.1.1 contamination_portion | 993 | 798 | 0.932 (0.914–0.946) | 0.000 | 0.988 (0.974–0.997) | 0.988 (0.974–0.997) |
| set_B_v2 | proGenomes 2.1 | MAGICC V5 predicted contamination | 993 | 798 | 0.960 (0.954–0.965) | 0.000 | 0.998 (0.995–1.000) | 0.998 (0.995–1.000) |
| set_B_v2 | proGenomes 2.1 | CheckM2 predicted contamination | 993 | 798 | 0.793 (0.762–0.821) | 0.000 | 0.996 (0.993–0.999) | 0.996 (0.992–0.999) |
| set_C_clean | proGenomes 2.1 | GUNC 1.1.1 CSS | 681 | 99 | 0.059 (-0.045–0.171) | 0.563 | 0.776 (0.572–0.943) | 0.635 (0.489–0.772) |
| set_C_clean | proGenomes 2.1 | GUNC 1.1.1 contamination_portion | 681 | 99 | 0.467 (0.350–0.570) | 0.000 | 0.978 (0.951–0.999) | 0.924 (0.840–0.992) |
| set_C_clean | proGenomes 2.1 | MAGICC V5 predicted contamination | 681 | 99 | 0.881 (0.851–0.907) | 0.000 | 0.991 (0.981–0.998) | 0.980 (0.959–0.995) |
| set_C_clean | proGenomes 2.1 | CheckM2 predicted contamination | 681 | 99 | 0.544 (0.482–0.600) | 0.000 | 0.927 (0.881–0.967) | 0.908 (0.866–0.945) |
| set_C_clean | GTDB r95 | GUNC 1.1.1 CSS | 880 | 100 | 0.100 (0.016–0.177) | 0.336 | 0.781 (0.672–0.880) | 0.637 (0.557–0.709) |
| set_C_clean | GTDB r95 | GUNC 1.1.1 contamination_portion | 880 | 100 | 0.837 (0.801–0.867) | 0.000 | 0.996 (0.990–0.999) | 0.994 (0.989–0.997) |
| set_C_clean | GTDB r95 | MAGICC V5 predicted contamination | 880 | 100 | 0.891 (0.859–0.916) | 0.000 | 0.967 (0.939–0.989) | 0.968 (0.951–0.985) |
| set_C_clean | GTDB r95 | CheckM2 predicted contamination | 880 | 100 | 0.582 (0.530–0.632) | 0.000 | 0.911 (0.871–0.945) | 0.911 (0.884–0.936) |
| set_D_clean | proGenomes 2.1 | GUNC 1.1.1 CSS | 945 | 98 | 0.067 (-0.014–0.149) | 0.521 | 0.693 (0.618–0.768) | 0.575 (0.518–0.634) |
| set_D_clean | proGenomes 2.1 | GUNC 1.1.1 contamination_portion | 945 | 98 | 0.848 (0.818–0.873) | 0.000 | 0.980 (0.962–0.994) | 0.988 (0.980–0.994) |
| set_D_clean | proGenomes 2.1 | MAGICC V5 predicted contamination | 945 | 98 | 0.929 (0.912–0.944) | 0.000 | 0.990 (0.976–0.997) | 0.989 (0.976–0.996) |
| set_D_clean | proGenomes 2.1 | CheckM2 predicted contamination | 945 | 98 | 0.689 (0.646–0.729) | 0.000 | 0.945 (0.925–0.964) | 0.932 (0.912–0.950) |
| set_D_clean | GTDB r95 | GUNC 1.1.1 CSS | 974 | 99 | 0.152 (0.073–0.230) | 0.148 | 0.718 (0.653–0.784) | 0.623 (0.578–0.675) |
| set_D_clean | GTDB r95 | GUNC 1.1.1 contamination_portion | 974 | 99 | 0.872 (0.846–0.894) | 0.000 | 0.996 (0.993–0.999) | 0.996 (0.992–0.998) |
| set_D_clean | GTDB r95 | MAGICC V5 predicted contamination | 974 | 99 | 0.931 (0.914–0.945) | 0.000 | 0.988 (0.974–0.997) | 0.987 (0.975–0.995) |
| set_D_clean | GTDB r95 | CheckM2 predicted contamination | 974 | 99 | 0.690 (0.649–0.729) | 0.000 | 0.943 (0.922–0.962) | 0.930 (0.910–0.949) |
| set_E | proGenomes 2.1 | GUNC 1.1.1 CSS | 986 | 777 | 0.550 (0.492–0.600) | 0.000 | 0.911 (0.885–0.936) | 0.867 (0.839–0.895) |
| set_E | proGenomes 2.1 | GUNC 1.1.1 contamination_portion | 986 | 777 | 0.852 (0.823–0.880) | 0.000 | 0.962 (0.947–0.975) | 0.961 (0.946–0.973) |
| set_E | proGenomes 2.1 | MAGICC V5 predicted contamination | 986 | 777 | 0.960 (0.951–0.967) | 0.000 | 0.995 (0.990–0.998) | 0.996 (0.993–0.999) |
| set_E | proGenomes 2.1 | CheckM2 predicted contamination | 986 | 777 | 0.843 (0.818–0.865) | 0.000 | 0.992 (0.987–0.996) | 0.987 (0.982–0.992) |
| set_E | GTDB r95 | GUNC 1.1.1 CSS | 992 | 779 | 0.617 (0.567–0.663) | 0.000 | 0.921 (0.898–0.945) | 0.877 (0.849–0.904) |
| set_E | GTDB r95 | GUNC 1.1.1 contamination_portion | 992 | 779 | 0.865 (0.838–0.889) | 0.000 | 0.974 (0.961–0.984) | 0.971 (0.958–0.980) |
| set_E | GTDB r95 | MAGICC V5 predicted contamination | 992 | 779 | 0.960 (0.952–0.967) | 0.000 | 0.995 (0.990–0.998) | 0.996 (0.993–0.999) |
| set_E | GTDB r95 | CheckM2 predicted contamination | 992 | 779 | 0.844 (0.819–0.866) | 0.000 | 0.992 (0.988–0.996) | 0.987 (0.982–0.992) |

### 4.3 stratum = `unpowered`

| Set | Database | score | n | clusters | Spearman ρ (95 % CI) | BH q | AUROC ≥5 % | AUROC ≥10 % |
|---|---|---|---|---|---|---|---|---|
| set_C_clean | proGenomes 2.1 | GUNC 1.1.1 CSS | 319 | 67 | 0.499 (0.372–0.620) | 0.000 | 0.908 (0.826–0.970) | 0.856 (0.798–0.908) |
| set_C_clean | proGenomes 2.1 | GUNC 1.1.1 contamination_portion | 319 | 67 | 0.324 (0.153–0.502) | 0.009 | 0.634 (0.506–0.789) | 0.660 (0.569–0.758) |
| set_C_clean | proGenomes 2.1 | MAGICC V5 predicted contamination | 319 | 67 | 0.852 (0.786–0.902) | 0.000 | 0.897 (0.852–0.944) | 0.929 (0.896–0.961) |
| set_C_clean | proGenomes 2.1 | CheckM2 predicted contamination | 319 | 67 | 0.696 (0.618–0.757) | 0.000 | 0.867 (0.811–0.910) | 0.864 (0.821–0.905) |
| set_C_clean | GTDB r95 | GUNC 1.1.1 CSS | 120 | 39 | 0.466 (0.259–0.644) | 0.003 | 0.855 (0.744–0.954) | 0.805 (0.688–0.901) |
| set_C_clean | GTDB r95 | GUNC 1.1.1 contamination_portion | 120 | 39 | 0.813 (0.643–0.933) | 0.000 | 0.949 (0.900–0.980) | 0.934 (0.878–0.974) |
| set_C_clean | GTDB r95 | MAGICC V5 predicted contamination | 120 | 39 | 0.856 (0.785–0.907) | 0.000 | 0.873 (0.799–0.941) | 0.919 (0.866–0.962) |
| set_C_clean | GTDB r95 | CheckM2 predicted contamination | 120 | 39 | 0.752 (0.681–0.804) | 0.000 | 0.878 (0.802–0.938) | 0.841 (0.757–0.907) |
| set_D_clean | proGenomes 2.1 | GUNC 1.1.1 CSS | 55 | 10 | 0.857 (0.730–0.922) | 0.002 | 0.992 (0.970–1.000) | 0.972 (0.948–1.000) |
| set_D_clean | proGenomes 2.1 | GUNC 1.1.1 contamination_portion | 55 | 10 | 0.247 (-0.142–0.696) | 0.505 | 0.798 (0.443–1.000) | 0.725 (0.436–0.912) |
| set_D_clean | proGenomes 2.1 | MAGICC V5 predicted contamination | 55 | 10 | 0.969 (0.926–0.981) | 0.000 | 0.944 (0.772–1.000) | 0.957 (0.886–1.000) |
| set_D_clean | proGenomes 2.1 | CheckM2 predicted contamination | 55 | 10 | 0.713 (0.498–0.885) | 0.024 | 0.836 (0.642–0.949) | 0.861 (0.722–0.955) |
| set_D_clean | GTDB r95 | GUNC 1.1.1 CSS | 26 | 6 | 0.921 (0.828–0.977) | 0.011 | 1.000 (1.000–1.000) | 0.995 (0.973–1.000) |
| set_D_clean | GTDB r95 | GUNC 1.1.1 contamination_portion | 26 | 6 | 0.575 (0.333–0.849) | 0.250 | 0.942 (0.867–1.000) | 0.933 (0.902–1.000) |
| set_D_clean | GTDB r95 | MAGICC V5 predicted contamination | 26 | 6 | 0.991 (0.979–1.000) | 0.000 | 0.986 (0.920–1.000) | 1.000 (1.000–1.000) |
| set_D_clean | GTDB r95 | CheckM2 predicted contamination | 26 | 6 | 0.660 (0.474–0.924) | 0.168 | 0.783 (0.550–1.000) | 0.829 (0.738–1.000) |

## 4b. The power effect, quantified — powered minus unpowered detection sensitivity

If the gap were a property of the *genomes* rather than of GUNC's reference coverage, MAGICC V5 and CheckM2 — which never consult a reference database at inference — would show the same gap on the same strata. They are therefore included as a specificity control. 95 % CI: unpaired cluster bootstrap resampling reference genomes within each stratum, 2000 resamples.

| Set | Database | thr | detector | n powered | n unpowered | sens. powered | sens. unpowered | Δ sensitivity (95 % CI) | spec. powered | spec. unpowered |
|---|---|---|---|---|---|---|---|---|---|---|
| set_C_clean | proGenomes 2.1 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 681 | 319 | 0.970 | 0.734 | **0.236** (0.152–0.327) | 0.733 | 0.865 |
| set_C_clean | proGenomes 2.1 | ≥5 % | MAGICC V5 (pred >= 5 %) | 681 | 319 | 0.994 | 0.947 | **0.047** (0.024–0.073) | 0.600 | 0.540 |
| set_C_clean | proGenomes 2.1 | ≥5 % | CheckM2 (pred >= 5 %) | 681 | 319 | 0.519 | 0.387 | **0.133** (0.057–0.207) | 1.000 | 1.000 |
| set_C_clean | proGenomes 2.1 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 681 | 319 | 0.975 | 0.789 | **0.186** (0.104–0.280) | 0.469 | 0.794 |
| set_C_clean | proGenomes 2.1 | ≥10 % | MAGICC V5 (pred >= 10 %) | 681 | 319 | 0.989 | 0.944 | **0.045** (0.015–0.081) | 0.625 | 0.662 |
| set_C_clean | proGenomes 2.1 | ≥10 % | CheckM2 (pred >= 10 %) | 681 | 319 | 0.176 | 0.187 | **-0.012** (-0.075–0.047) | 1.000 | 1.000 |
| set_C_clean | GTDB r95 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 880 | 120 | 0.999 | 1.000 | **-0.001** (-0.004–0.000) | 0.548 | 0.619 |
| set_C_clean | GTDB r95 | ≥5 % | MAGICC V5 (pred >= 5 %) | 880 | 120 | 0.986 | 0.929 | **0.057** (0.011–0.107) | 0.581 | 0.524 |
| set_C_clean | GTDB r95 | ≥5 % | CheckM2 (pred >= 5 %) | 880 | 120 | 0.491 | 0.384 | **0.107** (0.013–0.197) | 1.000 | 1.000 |
| set_C_clean | GTDB r95 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 880 | 120 | 1.000 | 1.000 | **0.000** (0.000–0.000) | 0.277 | 0.371 |
| set_C_clean | GTDB r95 | ≥10 % | MAGICC V5 (pred >= 10 %) | 880 | 120 | 0.984 | 0.906 | **0.078** (0.021–0.142) | 0.646 | 0.657 |
| set_C_clean | GTDB r95 | ≥10 % | CheckM2 (pred >= 10 %) | 880 | 120 | 0.179 | 0.176 | **0.003** (-0.080–0.094) | 1.000 | 1.000 |
| set_D_clean | proGenomes 2.1 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 945 | 55 | 1.000 | 0.880 | **0.120** (0.030–0.182) | 0.316 | 1.000 |
| set_D_clean | proGenomes 2.1 | ≥5 % | MAGICC V5 (pred >= 5 %) | 945 | 55 | 0.989 | 0.980 | **0.009** (-0.017–0.043) | 0.860 | 0.800 |
| set_D_clean | proGenomes 2.1 | ≥5 % | CheckM2 (pred >= 5 %) | 945 | 55 | 0.702 | 0.380 | **0.322** (0.123–0.510) | 1.000 | 1.000 |
| set_D_clean | proGenomes 2.1 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 945 | 55 | 1.000 | 0.935 | **0.065** (0.016–0.103) | 0.173 | 0.889 |
| set_D_clean | proGenomes 2.1 | ≥10 % | MAGICC V5 (pred >= 10 %) | 945 | 55 | 0.980 | 0.913 | **0.067** (0.010–0.105) | 0.913 | 0.667 |
| set_D_clean | proGenomes 2.1 | ≥10 % | CheckM2 (pred >= 10 %) | 945 | 55 | 0.442 | 0.174 | **0.268** (0.076–0.423) | 1.000 | 1.000 |
| set_D_clean | GTDB r95 | ≥5 % | GUNC 1.1.1 (pass.GUNC == False) | 974 | 26 | 1.000 | 1.000 | **0.000** (0.000–0.000) | 0.322 | 1.000 |
| set_D_clean | GTDB r95 | ≥5 % | MAGICC V5 (pred >= 5 %) | 974 | 26 | 0.989 | 0.957 | **0.033** (-0.016–0.147) | 0.848 | 1.000 |
| set_D_clean | GTDB r95 | ≥5 % | CheckM2 (pred >= 5 %) | 974 | 26 | 0.693 | 0.348 | **0.345** (-0.056–0.524) | 1.000 | 1.000 |
| set_D_clean | GTDB r95 | ≥10 % | GUNC 1.1.1 (pass.GUNC == False) | 974 | 26 | 1.000 | 1.000 | **0.000** (0.000–0.000) | 0.176 | 0.600 |
| set_D_clean | GTDB r95 | ≥10 % | MAGICC V5 (pred >= 10 %) | 974 | 26 | 0.979 | 0.857 | **0.122** (0.028–0.240) | 0.889 | 1.000 |
| set_D_clean | GTDB r95 | ≥10 % | CheckM2 (pred >= 10 %) | 974 | 26 | 0.435 | 0.143 | **0.292** (0.009–0.421) | 1.000 | 1.000 |

**How to read this table honestly.** Under proGenomes 2.1 on `set_C_clean`, GUNC loses Δ +0.236 of sensitivity in the unpowered stratum while the reference-free MAGICC V5 loses only Δ +0.047 on the identical genomes — a five-fold difference that is hard to explain as genome difficulty. But CheckM2, also reference-free at inference, loses Δ +0.133 on `set_C_clean` and Δ +0.322 on `set_D_clean`, so the unpowered stratum **is** intrinsically somewhat harder and the reference-coverage argument must not be overstated from these columns alone.

The decisive observation is the database swap. Moving to GTDB r95 collapses GUNC's power gap to essentially zero (Δ −0.001 on `set_C_clean`, Δ 0.000 on `set_D_clean`) while MAGICC V5's and CheckM2's gaps persist on the same genomes. A change that touches only the reference database removes GUNC's gap and leaves the reference-free tools' gaps intact: **GUNC's power gap was reference coverage; the residual gap in the other tools is genome difficulty.**


## 5. Database sensitivity — proGenomes 2.1 vs GTDB r95, paired on identical genomes

This section answers protocol §4.4c item 4 at full scale (it was previously a 20-genome CPR pilot). Δ = GTDB r95 − proGenomes 2.1; 95 % CI by cluster bootstrap over reference genomes. Reported for two strata: `all_paired` (every genome in both arms) and `powered_both_dbs` (only genomes GUNC was powered to adjudicate **under both databases**) — in the latter, a verdict change cannot be explained by one arm simply having no close reference.

**Correction to the 20-genome pilot.** The pilot reported median RRS 0.575 → 0.940 and median AA identity 0.850 → 0.980 on `set_C_clean`. Those 20 genomes reproduce bit-identically inside the full run, so the pilot was not wrong about them — it was an unrepresentative sample. Across all 1,000 genomes the gain is far smaller (median RRS 0.56 → 0.64, median AA identity 0.77 → 0.80). The powered-fraction gain held up (12/20 → 17/20 in the pilot; 681 → 880 of 1,000 at full scale). **The full-scale numbers supersede the pilot.**


### 5.1 stratum = `all_paired`

| Set | n paired | metric | proGenomes 2.1 | GTDB r95 | Δ (95 % CI) |
|---|---|---|---|---|---|
| set_C_clean | 1000 | fraction powered | 0.681 | 0.880 | 0.199 (0.156–0.248) |
| set_C_clean | 1000 | fraction hard-unpowered (CSS forcibly zeroed) | 0.001 | 0.000 | -0.001 (-0.003–0.000) |
| set_C_clean | 1000 | median reference_representation_score | 0.560 | 0.640 | 0.080 (0.060–0.100) |
| set_C_clean | 1000 | median genes_retained_index | 0.740 | 0.830 | 0.090 (0.070–0.110) |
| set_C_clean | 1000 | median mean AA hit identity | 0.770 | 0.800 | 0.030 (0.020–0.050) |
| set_C_clean | 1000 | median n_genes_mapped | 992.500 | 1021.500 | 29.000 (15.500–37.500) |
| set_C_clean | 1000 | overall GUNC fail rate | 0.862 | 0.969 | 0.107 (0.072–0.145) |
| set_C_clean | 1000 | median CSS | 0.800 | 1.000 | 0.200 (0.140–0.250) |
| set_C_clean | 1000 | Spearman CSS vs true contamination | 0.321 | 0.200 | -0.121 (-0.205–-0.041) |
| set_C_clean | 1000 | detection sensitivity at >=5 % contamination | 0.900 | 0.999 | 0.099 (0.067–0.135) |
| set_C_clean | 1000 | detection specificity at >=5 % contamination | 0.827 | 0.577 | -0.250 (-0.383–-0.106) |
| set_C_clean | 1000 | detection sensitivity at >=10 % contamination | 0.923 | 1.000 | 0.077 (0.046–0.110) |
| set_C_clean | 1000 | detection specificity at >=10 % contamination | 0.690 | 0.310 | -0.380 (-0.495–-0.274) |
| set_D_clean | 1000 | fraction powered | 0.945 | 0.974 | 0.029 (0.010–0.054) |
| set_D_clean | 1000 | fraction hard-unpowered (CSS forcibly zeroed) | 0.000 | 0.001 | 0.001 (0.000–0.003) |
| set_D_clean | 1000 | median reference_representation_score | 0.840 | 0.920 | 0.080 (0.060–0.110) |
| set_D_clean | 1000 | median genes_retained_index | 0.940 | 0.970 | 0.030 (0.020–0.030) |
| set_D_clean | 1000 | median mean AA hit identity | 0.890 | 0.950 | 0.060 (0.040–0.080) |
| set_D_clean | 1000 | median n_genes_mapped | 2469.500 | 2515.500 | 46.000 (19.000–63.500) |
| set_D_clean | 1000 | overall GUNC fail rate | 0.971 | 0.978 | 0.007 (0.000–0.015) |
| set_D_clean | 1000 | median CSS | 1.000 | 1.000 | 0.000 (0.000–0.000) |
| set_D_clean | 1000 | Spearman CSS vs true contamination | 0.103 | 0.199 | 0.096 (0.032–0.162) |
| set_D_clean | 1000 | detection sensitivity at >=5 % contamination | 0.994 | 1.000 | 0.006 (0.001–0.014) |
| set_D_clean | 1000 | detection specificity at >=5 % contamination | 0.371 | 0.355 | -0.016 (-0.087–0.054) |
| set_D_clean | 1000 | detection sensitivity at >=10 % contamination | 0.997 | 1.000 | 0.003 (0.000–0.008) |
| set_D_clean | 1000 | detection specificity at >=10 % contamination | 0.230 | 0.195 | -0.035 (-0.095–0.017) |
| set_E | 1000 | fraction powered | 0.986 | 0.992 | 0.006 (0.001–0.012) |
| set_E | 1000 | fraction hard-unpowered (CSS forcibly zeroed) | 0.000 | 0.000 | 0.000 (0.000–0.000) |
| set_E | 1000 | median reference_representation_score | 0.900 | 0.940 | 0.040 (0.030–0.050) |
| set_E | 1000 | median genes_retained_index | 0.960 | 0.970 | 0.010 (0.010–0.020) |
| set_E | 1000 | median mean AA hit identity | 0.950 | 0.970 | 0.020 (0.020–0.020) |
| set_E | 1000 | median n_genes_mapped | 4011.500 | 4043.500 | 32.000 (15.488–76.000) |
| set_E | 1000 | overall GUNC fail rate | 0.761 | 0.772 | 0.011 (0.002–0.021) |
| set_E | 1000 | median CSS | 1.000 | 1.000 | 0.000 (0.000–0.010) |
| set_E | 1000 | Spearman CSS vs true contamination | 0.559 | 0.622 | 0.063 (0.037–0.090) |
| set_E | 1000 | detection sensitivity at >=5 % contamination | 0.960 | 0.973 | 0.013 (0.006–0.022) |
| set_E | 1000 | detection specificity at >=5 % contamination | 0.829 | 0.825 | -0.004 (-0.032–0.024) |
| set_E | 1000 | detection sensitivity at >=10 % contamination | 0.960 | 0.972 | 0.013 (0.005–0.021) |
| set_E | 1000 | detection specificity at >=10 % contamination | 0.747 | 0.740 | -0.007 (-0.034–0.018) |

### 5.2 stratum = `powered_both_dbs`

| Set | n paired | metric | proGenomes 2.1 | GTDB r95 | Δ (95 % CI) |
|---|---|---|---|---|---|
| set_C_clean | 663 | fraction powered | 1.000 | 1.000 | 0.000 (0.000–0.000) |
| set_C_clean | 663 | fraction hard-unpowered (CSS forcibly zeroed) | 0.000 | 0.000 | 0.000 (0.000–0.000) |
| set_C_clean | 663 | median reference_representation_score | 0.620 | 0.690 | 0.070 (0.050–0.090) |
| set_C_clean | 663 | median genes_retained_index | 0.790 | 0.860 | 0.070 (0.060–0.100) |
| set_C_clean | 663 | median mean AA hit identity | 0.810 | 0.830 | 0.020 (0.010–0.030) |
| set_C_clean | 663 | median n_genes_mapped | 1063.000 | 1079.000 | 16.000 (10.000–34.000) |
| set_C_clean | 663 | overall GUNC fail rate | 0.956 | 0.986 | 0.030 (0.014–0.048) |
| set_C_clean | 663 | median CSS | 0.880 | 1.000 | 0.120 (0.030–0.165) |
| set_C_clean | 663 | Spearman CSS vs true contamination | 0.055 | -0.009 | -0.064 (-0.172–0.029) |
| set_C_clean | 663 | detection sensitivity at >=5 % contamination | 0.971 | 0.999 | 0.028 (0.013–0.046) |
| set_C_clean | 663 | detection specificity at >=5 % contamination | 0.714 | 0.571 | -0.143 (-0.375–0.000) |
| set_C_clean | 663 | detection sensitivity at >=10 % contamination | 0.976 | 1.000 | 0.024 (0.010–0.040) |
| set_C_clean | 663 | detection specificity at >=10 % contamination | 0.452 | 0.290 | -0.161 (-0.364–0.029) |
| set_D_clean | 943 | fraction powered | 1.000 | 1.000 | 0.000 (0.000–0.000) |
| set_D_clean | 943 | fraction hard-unpowered (CSS forcibly zeroed) | 0.000 | 0.000 | 0.000 (0.000–0.000) |
| set_D_clean | 943 | median reference_representation_score | 0.850 | 0.920 | 0.070 (0.050–0.100) |
| set_D_clean | 943 | median genes_retained_index | 0.950 | 0.970 | 0.020 (0.020–0.030) |
| set_D_clean | 943 | median mean AA hit identity | 0.900 | 0.950 | 0.050 (0.030–0.080) |
| set_D_clean | 943 | median n_genes_mapped | 2523.000 | 2566.000 | 43.000 (17.000–64.000) |
| set_D_clean | 943 | overall GUNC fail rate | 0.982 | 0.981 | -0.001 (-0.005–0.002) |
| set_D_clean | 943 | median CSS | 1.000 | 1.000 | 0.000 (0.000–0.000) |
| set_D_clean | 943 | Spearman CSS vs true contamination | 0.063 | 0.141 | 0.078 (-0.001–0.157) |
| set_D_clean | 943 | detection sensitivity at >=5 % contamination | 1.000 | 1.000 | 0.000 (0.000–0.000) |
| set_D_clean | 943 | detection specificity at >=5 % contamination | 0.304 | 0.321 | 0.018 (-0.039–0.086) |
| set_D_clean | 943 | detection sensitivity at >=10 % contamination | 1.000 | 1.000 | 0.000 (0.000–0.000) |
| set_D_clean | 943 | detection specificity at >=10 % contamination | 0.165 | 0.175 | 0.010 (-0.021–0.045) |
| set_E | 986 | fraction powered | 1.000 | 1.000 | 0.000 (0.000–0.000) |
| set_E | 986 | fraction hard-unpowered (CSS forcibly zeroed) | 0.000 | 0.000 | 0.000 (0.000–0.000) |
| set_E | 986 | median reference_representation_score | 0.900 | 0.940 | 0.040 (0.030–0.050) |
| set_E | 986 | median genes_retained_index | 0.960 | 0.970 | 0.010 (0.010–0.020) |
| set_E | 986 | median mean AA hit identity | 0.950 | 0.970 | 0.020 (0.020–0.020) |
| set_E | 986 | median n_genes_mapped | 4043.000 | 4091.000 | 48.000 (13.000–72.000) |
| set_E | 986 | overall GUNC fail rate | 0.767 | 0.777 | 0.010 (0.001–0.020) |
| set_E | 986 | median CSS | 1.000 | 1.000 | 0.000 (0.000–0.005) |
| set_E | 986 | Spearman CSS vs true contamination | 0.550 | 0.615 | 0.065 (0.037–0.094) |
| set_E | 986 | detection sensitivity at >=5 % contamination | 0.962 | 0.974 | 0.012 (0.004–0.021) |
| set_E | 986 | detection specificity at >=5 % contamination | 0.828 | 0.824 | -0.004 (-0.033–0.027) |
| set_E | 986 | detection sensitivity at >=10 % contamination | 0.962 | 0.973 | 0.011 (0.004–0.020) |
| set_E | 986 | detection specificity at >=10 % contamination | 0.744 | 0.736 | -0.007 (-0.035–0.019) |

### 5.3 Verdict and power transitions when the database is swapped

Every genome whose GUNC verdict changes between the two databases, classified against the ≥5 % ground truth. A database that merely fails more genomes is not thereby better — what matters is whether the extra failures are true positives.

| Set | n paired | same verdict | pass→fail (GTDB) | of which truly ≥5 % | of which clean (<5 %) | fail→pass (GTDB) | of which truly ≥5 % (lost) | net true positives | net false positives | unpowered→powered | powered→unpowered |
|---|---|---|---|---|---|---|---|---|---|---|---|
| set_C_clean | 1000 | 887 | **110** | 95 | 15 | 3 | 1 | **+94** | +13 | 217 | 18 |
| set_D_clean | 1000 | 989 | **9** | 6 | 3 | 2 | 0 | **+6** | +1 | 31 | 2 |
| set_E | 1000 | 977 | **17** | 10 | 7 | 6 | 0 | **+10** | +1 | 6 | 0 |

**The trade-off, stated explicitly and not resolved in favour of either database.** GTDB r95 buys reference coverage — 1,131 CPR and 1,672 archaeal references against proGenomes 2.1's 104 and 457 — and that is what moves genomes into the powered stratum. It buys it with 14,566 GenBank-prefixed, largely MAG-derived reference genomes, which may themselves be contaminated; a "fail" against a contaminated reference is not the same evidence as a "fail" against a finished genome. proGenomes 2.1 is the cleaner reference set and is near-blind on `set_C_clean`, which is entirely Patescibacteriota. **Neither is strictly superior. Both are reported; the comparison stands as the result.**


### 5.4 The database hypothesis tested directly on the WS4.1 real-genome controls

The synthetic sets above cannot settle *why* GUNC missed the eight Kraken2-"confirmed" contaminated MAGs of §4.4c, because those are real novel-lineage genomes. Those 14 control genomes (8 putative positives, 6 finished pure cultures) were therefore re-scored under both databases. If poor reference coverage were the explanation, GTDB r95 should power GUNC up **and change the calls**. It does the first and not the second.

| Control arm | Database | n | powered | GUNC fails | fails within powered | median RRS | median AA id | median CSS | max CSS |
|---|---|---|---|---|---|---|---|---|---|
| positive | proGenomes 2.1 | 8 | **1/8** | 0 | 0 | 0.455 | 0.575 | 0.075 | 0.320 |
| positive | GTDB r95 | 8 | **5/8** | 0 | 0 | 0.510 | 0.695 | 0.125 | 0.250 |
| negative | proGenomes 2.1 | 6 | **6/6** | 0 | 0 | 0.975 | 0.985 | 0.000 | 0.000 |
| negative | GTDB r95 | 6 | **6/6** | 0 | 0 | 0.970 | 0.980 | 0.000 | 0.000 |

Swapping to GTDB r95 raises the powered count on the putative positives from **1/8 to 5/8** — the database change did exactly what it was supposed to do — and GUNC still flags **0/8**, with every CSS at or below 0.25 against a 0.45 failure threshold. The 6/6 finished pure cultures continue to pass at CSS 0.000 under both databases, so this is not an insensitive install. **The proGenomes-2.1 result was therefore not a database-capacity artefact**, which removes the last alternative explanation available to the withdrawn Kraken2 claim (§4.4c). This table belongs to §4.4c/WS4.3 and is reproduced here only because it is a database-sensitivity result; it is not part of the WS4.2 benchmark comparison.


## 6. Runs included

| Set | Database | normalized TSV |
|---|---|---|
| set_A_v2 | proGenomes 2.1 | `results/revision/gunc/runs/set_A_v2/progenomes_2.1/gunc_normalized.tsv` |
| set_B_v2 | proGenomes 2.1 | `results/revision/gunc/runs/set_B_v2/progenomes_2.1/gunc_normalized.tsv` |
| set_C_clean | proGenomes 2.1 | `results/revision/gunc/runs/set_C_clean/progenomes_2.1/gunc_normalized.tsv` |
| set_C_clean | GTDB r95 | `results/revision/gunc/runs/set_C_clean/gtdb_95/gunc_normalized.tsv` |
| set_D_clean | proGenomes 2.1 | `results/revision/gunc/runs/set_D_clean/progenomes_2.1/gunc_normalized.tsv` |
| set_D_clean | GTDB r95 | `results/revision/gunc/runs/set_D_clean/gtdb_95/gunc_normalized.tsv` |
| set_E | proGenomes 2.1 | `results/revision/gunc/runs/set_E/progenomes_2.1/gunc_normalized.tsv` |
| set_E | GTDB r95 | `results/revision/gunc/runs/set_E/gtdb_95/gunc_normalized.tsv` |

## 7. Notes and caveats

* set_A_v2/proGenomes 2.1: rank/AUROC statistics omitted — true contamination is uniformly 0 % by construction, so Spearman ρ and AUROC are undefined (R1-m19 zero-true-variance convention). Pass/fail and the power audit are still reported: this set is a pure false-positive (specificity) test.
