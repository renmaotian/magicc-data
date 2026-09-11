# GORG-Tropics SAG denominator audit

The ENA manifest contains 12,715 assemblies. Of these, 3,160
(24.9%) match the local GTDB metadata. All 3,160
matched records have published CheckM2 completeness ≥50%. Their median CheckM2
completeness is 75.41%, median contamination
0.05% and maximum contamination
3.11%. These are statistics of the **GTDB-matched
subset**, not of all 12,715 GORG assemblies.

The historical 3,151 count is reproduced exactly by retaining only
GTDB records with `ncbi_bioproject == PRJEB33281`. Their completeness median is
75.39% and 278
have completeness ≥90%. The stronger accession/linked-GenBank join recovers
9 additional RefSeq assemblies recorded under PRJNA224116; these
records and accessions are provided in additional_refseq_linked_records.tsv.

The prior claim that the other 9,555 assemblies necessarily lie
below MAGICC's 50% floor was an unsupported inference from GTDB absence. GTDB
applies multiple inclusion criteria, including contamination, quality score,
marker recovery and assembly quality; non-inclusion does not identify which
criterion was failed. See [GTDB methods](https://gtdb.ecogenomic.org/methods) and
[GTDB release10 primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12807784/).

Corrected table wording: “GORG-Tropics assemblies matching local GTDB metadata:
3,160/12,715 (24.9%); all matched records have
CheckM2 completeness ≥50%. The ≥50% fraction and median completeness of the full
GORG collection were not established by this screen.” Remove “roughly three
quarters cannot be scored at all.” Replace “Median ... of that collection” with
“Published CheckM2 median ... among the 3,160 GTDB-matched assemblies.”

This is a metadata feasibility screen. No new MAGICC SAG predictions or
reference-anchored accuracy evaluation were performed. The previously identified
453 conspecific-reference candidates remain potential material, not validated
SAG performance. “Scoreable” should describe the intended model domain and must
not imply the CLI refuses incomplete inputs or that tool estimates are truth.

The independent general GTDB annotation screen contains 1,596 records labeled
`derived from single cell`, with 1,595 at CheckM2
completeness ≥50% and median completeness 75.07%.
This is a different, non-exhaustive metadata category and should not be conflated
with the GORG BioProject membership screen.

Reproduction: `python scripts/249_audit_gorg_sag_source.py`. Exact selected records,
summary and input SHA256 hashes are stored alongside this report.
