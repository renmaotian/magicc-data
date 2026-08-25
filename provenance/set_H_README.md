# WS1.11 (R1-m13) — provenance audit for `set_H_ncbi`

Generated 2026-08-01T18:46:14.131627+00:00 by `data_generating_scripts/188_ws1_11_provenance_audit.py`,
which imports the normalisation and GCA↔GCF cross-map code of
`data_generating_scripts/74_provenance_audit.py` directly, so the two audits cannot drift apart.

## Verdict: **PASS**

| universe | samples (n=4000) | unique reference genomes (n=400) |
|---|---|---|
| training split | 0 | 0 |
| validation split | 0 | 0 |
| test split | 350 | 35 |
| 2,000-genome 9-mer feature-selection set | 0 | 0 |
| 277,183-genome CheckM2-filtered curation pool | 2000 | 200 |

The last row is the point of the experiment. The **H_fail** arm
(2000 samples / 200 references)
has 0 members inside that pool: these
are exactly the genomes the project's CheckM2-based curation removed, so no MAGICC model
and no previous MAGICC benchmark has ever seen them or anything selected the way they
were.

Counts are identical under all three normalisations (raw GTDB string, strict
version-stripped string, GCA↔GCF cross-mapped canonical key); the cross-mapped column is
the one to trust. Cross-map: 277,183 rows →
503,511 accession strings →
277,183 canonical assemblies
(1.817 strings/assembly,
0 inconsistencies).

## Contaminants

12,075 contamination events over
6,853 unique genomes, **all** from the held-out
test split (0 from train,
0 from val,
0 from no split), and
0 of them share the dominant's phylum —
identical to the contaminant provenance of `set_C_clean` / `set_D_clean`, because the
contaminant pool is deliberately unchanged.

## Label domain (protocol §4.4a)

0 of 4,000 samples violate `contamination% ≤ completeness%`, the
constraint the V5 training distribution enforces. Out-of-domain samples, if any, are
reported separately in the analysis.

## Files

| file | content |
|---|---|
| `overlap_summary.tsv` | per-sample and per-genome overlap counts against every universe, under three normalisations |
| `set_H_ncbi_dominants.txt` | the 400 reference genomes with taxonomy, arm, CheckM2 scores and split membership |
| `contaminants.txt` | one row per contamination event |
| `sha256_manifest.txt` | 4413 files: every generated FASTA, every reference FASTA, metadata, labels, the frozen ONNX model, normalisation parameters, the 9-mer list, the split accession lists |
| `audit_summary.json` | machine-readable form of the above |

`models/magicc_v5.onnx` SHA256 `b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096` — matches the frozen V5 model
(`b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`): **True**. No model was retrained or modified for
WS1.11; it is an evaluation-only workstream.
