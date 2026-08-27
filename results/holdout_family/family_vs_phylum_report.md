# WS1.9 - leave-FAMILY-out holdout: does MAGICC generalize to a novel family inside a seen phylum?

**Holdout models are validation artifacts only. The released MAGICC model (`models/magicc_v5.onnx`) remains trained on all data.**

## Design

- Panel: 25 families in 6 evaluation groups, drawn from Bacteroidota, Campylobacterota, Halobacteriota, Patescibacteriota.
- **Every held-out family's parent phylum remains in training**: all 110 phyla survive; phyla eliminated: none.
- Training pool removed: 5,558/79,948 = **6.952%** (WS1.6 phylum panel removed 19.64%).
- Reduced-genome pool: V5 definition used verbatim, no adaptation needed (63.52% survives).
- Training data matched to V5 exactly (1M/100k/100k; 15/15/30/30/5/5%), so there is no sample-size confound. Seed 42.
- Panel families excluded from BOTH the dominant and the contaminant pool; evaluation dominants come from the held-out TEST split.

## Primary comparison (paired, identical samples)

All three models score the same genomes:

| model | saw the family? | saw the phylum? |
|---|---|---|
| production V5 | yes | yes |
| family-holdout (WS1.9) | **no** | yes |
| phylum-holdout (WS1.6) | **no** | **no** |

### Completeness

```
                             group     parent_phylum  n_refs  comp_did_family comp_ci95_family  comp_q_family_bh  comp_did_phylum  comp_attenuation comp_ci95_attenuation  comp_pct_of_phylum_effect  comp_q_attenuation_bh
       Bacteroidota_Muribaculaceae      Bacteroidota     100           3.8866 [3.1856, 4.6558]            0.0005           6.3697            2.4831      [1.7115, 3.1749]                       61.0                0.00060
    Bacteroidota_Flavobacteriaceae      Bacteroidota     100           4.1652 [3.5327, 4.8496]            0.0005           7.3191            3.1539      [2.2934, 4.0927]                       56.9                0.00060
Campylobacterota_Helicobacteraceae  Campylobacterota     100           5.2007 [4.5708, 5.8153]            0.0005           4.8062           -0.3945     [-1.0532, 0.2391]                      108.2                0.24873
  Campylobacterota_Arcobacteraceae  Campylobacterota      40           5.0070 [4.1324, 5.8967]            0.0005           7.9759            2.9688       [1.834, 4.0137]                       62.8                0.00060
         Halobacteriota_halophilic    Halobacteriota      28           2.9483 [1.9189, 3.9739]            0.0005           4.5320            1.5837       [0.6834, 2.423]                       65.1                0.00060
        Patescibacteriota_families Patescibacteriota      66           5.0284 [3.5416, 6.7233]            0.0005          24.6008           19.5724    [17.8695, 21.2659]                       20.4                0.00060
```

### Contamination

```
                             group     parent_phylum  n_refs  cont_did_family   cont_ci95_family  cont_q_family_bh  cont_did_phylum  cont_attenuation cont_ci95_attenuation  cont_pct_of_phylum_effect  cont_q_attenuation_bh
       Bacteroidota_Muribaculaceae      Bacteroidota     100           3.1883    [2.5359, 3.871]            0.0005           4.9158            1.7274         [1.01, 2.404]                       64.9                 0.0006
    Bacteroidota_Flavobacteriaceae      Bacteroidota     100           4.2396   [3.1867, 5.3094]            0.0005          15.1791           10.9394     [9.2732, 12.6713]                       27.9                 0.0006
Campylobacterota_Helicobacteraceae  Campylobacterota     100          12.3525 [11.2308, 13.4781]            0.0005          14.7472            2.3946      [1.5042, 3.3307]                       83.8                 0.0006
  Campylobacterota_Arcobacteraceae  Campylobacterota      40           1.8182   [1.0761, 2.5673]            0.0005           2.0211            0.2029     [-0.4729, 0.8598]                       90.0                 0.5520
         Halobacteriota_halophilic    Halobacteriota      28           1.5338     [0.71, 2.3377]            0.0005           3.2522            1.7184      [0.9779, 2.4504]                       47.2                 0.0006
        Patescibacteriota_families Patescibacteriota      66           3.6593   [2.3706, 5.1399]            0.0005          13.7339           10.0746     [8.6269, 11.5799]                       26.6                 0.0006
```

Common control: 800/1000 in_distribution samples (80 references), restricted to dominant phyla outside the WS1.6 panel so the control is valid for both holdout models. Control deltas vs V5: {'familyHO_comp': 0.0348, 'familyHO_cont': -0.2353, 'phylumHO_comp': 0.0135, 'phylumHO_cont': -0.3147}.

## Files

- `family_vs_phylum_did.tsv` - the deliverable
- `three_model_head_to_head.tsv`
- `cross_experiment_did_summary.tsv`
- `lineage_novelty_effect_did.tsv` - WS1.9 standalone DiD
- `head_to_head_by_group.tsv`, `stratified_error_by_band.tsv`, `mimag_threshold_by_group.tsv`, `mimag_confusion_by_group.tsv`, `per_reference_errors.tsv`, `sub_phylum_breakdown.tsv`
