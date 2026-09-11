#!/usr/bin/env python3
"""Audit the GORG-Tropics screening denominator; no SAG prediction is generated."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision/holdout_resubmission5/sag_source_audit'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def accession_base(s):return s.str.replace(r'^(RS_|GB_)','',regex=True).str.replace(r'\.\d+$','',regex=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ena=ROOT/'data/real_data/gorg_tropics/ena_PRJEB33281_assemblies.tsv'
    manifest=pd.read_csv(ena,sep='\t');wanted=set(accession_base(manifest.accession));assert len(wanted)==len(manifest)
    rows=[];sag=[];inputs=[ena]
    columns=['accession','ncbi_genbank_assembly_accession','ncbi_bioproject','checkm2_completeness','checkm2_contamination','gtdb_taxonomy','ncbi_genome_category']
    for filename in ['bac120_metadata.tsv.gz','ar53_metadata.tsv.gz']:
        path=ROOT/'data/gtdb'/filename;inputs.append(path)
        for chunk in pd.read_csv(path,sep='\t',usecols=columns,chunksize=50000,low_memory=False):
            match=accession_base(chunk.accession).isin(wanted)|accession_base(chunk.ncbi_genbank_assembly_accession).isin(wanted)
            rows.append(chunk[match].copy());sag.append(chunk[chunk.ncbi_genome_category.eq('derived from single cell')].copy())
    d=pd.concat(rows,ignore_index=True);s=pd.concat(sag,ignore_index=True)
    # ENA supplies unversioned assembly accessions. Ensure the join counts each
    # ENA assembly once, even when the GTDB representative is a linked RefSeq.
    direct=accession_base(d.accession);linked_base=accession_base(d.ncbi_genbank_assembly_accession)
    d['matched_ena_accession']=direct.where(direct.isin(wanted),linked_base)
    assert d.matched_ena_accession.isin(wanted).all() and d.matched_ena_accession.is_unique
    literal=d[d.ncbi_bioproject.eq('PRJEB33281')]
    linked=d[~d.ncbi_bioproject.eq('PRJEB33281')]
    d.to_csv(OUT/'gorg_gtdb_matched_metadata.tsv',sep='\t',index=False)
    linked.to_csv(OUT/'additional_refseq_linked_records.tsv',sep='\t',index=False)
    s.to_csv(OUT/'all_gtdb_single_cell_metadata.tsv',sep='\t',index=False)
    summary={'ena_gorg_assemblies':len(manifest),'gorg_matching_gtdb_records':len(d),
             'matched_unique_ena_accessions':d.matched_ena_accession.nunique(),
             'join_scope':'ENA manifest contains unversioned assembly accessions; each matched assembly is counted once, including linked RefSeq aliases.',
             'fraction_of_ena_present_in_gtdb':len(d)/len(manifest),
             'matched_checkm2_completeness_ge50':int(d.checkm2_completeness.ge(50).sum()),
             'matched_checkm2_completeness_median':float(d.checkm2_completeness.median()),
             'matched_checkm2_completeness_min':float(d.checkm2_completeness.min()),
             'matched_checkm2_completeness_ge90':int(d.checkm2_completeness.ge(90).sum()),
             'matched_checkm2_contamination_median':float(d.checkm2_contamination.median()),
             'matched_checkm2_contamination_max':float(d.checkm2_contamination.max()),
             'gorg_not_present_in_gtdb':len(manifest)-len(d),
             'historical_literal_bioproject_count':len(literal),
             'historical_literal_bioproject_comp_median':float(literal.checkm2_completeness.median()),
             'historical_literal_bioproject_comp_ge90':int(literal.checkm2_completeness.ge(90).sum()),
             'additional_linked_refseq_records':len(linked),
             'whole_cohort_below50_count':None,
             'whole_cohort_completeness_median':None,
             'absence_reason':'Cannot infer completeness below50% from absence in GTDB; inclusion has multiple criteria.',
             'all_gtdb_explicit_single_cell_count':len(s),
             'all_gtdb_explicit_single_cell_checkm2_ge50':int(s.checkm2_completeness.ge(50).sum()),
             'all_gtdb_explicit_single_cell_comp_median':float(s.checkm2_completeness.median()),
             'new_SAG_accuracy_evaluation_performed':False,
             'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in inputs]}
    (OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'audit_report.md').write_text(f'''# GORG-Tropics SAG denominator audit

The ENA manifest contains {len(manifest):,} assemblies. Of these, {len(d):,}
({100*len(d)/len(manifest):.1f}%) match the local GTDB metadata. All {int(d.checkm2_completeness.ge(50).sum()):,}
matched records have published CheckM2 completeness ≥50%. Their median CheckM2
completeness is {d.checkm2_completeness.median():.2f}%, median contamination
{d.checkm2_contamination.median():.2f}% and maximum contamination
{d.checkm2_contamination.max():.2f}%. These are statistics of the **GTDB-matched
subset**, not of all {len(manifest):,} GORG assemblies.

The historical {len(literal):,} count is reproduced exactly by retaining only
GTDB records with `ncbi_bioproject == PRJEB33281`. Their completeness median is
{literal.checkm2_completeness.median():.2f}% and {int(literal.checkm2_completeness.ge(90).sum())}
have completeness ≥90%. The stronger accession/linked-GenBank join recovers
{len(linked)} additional RefSeq assemblies recorded under PRJNA224116; these
records and accessions are provided in additional_refseq_linked_records.tsv.

The prior claim that the other {len(manifest)-len(d):,} assemblies necessarily lie
below MAGICC's 50% floor was an unsupported inference from GTDB absence. GTDB
applies multiple inclusion criteria, including contamination, quality score,
marker recovery and assembly quality; non-inclusion does not identify which
criterion was failed. See [GTDB methods](https://gtdb.ecogenomic.org/methods) and
[GTDB release10 primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12807784/).

Corrected table wording: “GORG-Tropics assemblies matching local GTDB metadata:
{len(d):,}/{len(manifest):,} ({100*len(d)/len(manifest):.1f}%); all matched records have
CheckM2 completeness ≥50%. The ≥50% fraction and median completeness of the full
GORG collection were not established by this screen.” Remove “roughly three
quarters cannot be scored at all.” Replace “Median ... of that collection” with
“Published CheckM2 median ... among the {len(d):,} GTDB-matched assemblies.”

This is a metadata feasibility screen. No new MAGICC SAG predictions or
reference-anchored accuracy evaluation were performed. The previously identified
453 conspecific-reference candidates remain potential material, not validated
SAG performance. “Scoreable” should describe the intended model domain and must
not imply the CLI refuses incomplete inputs or that tool estimates are truth.

The independent general GTDB annotation screen contains {len(s):,} records labeled
`derived from single cell`, with {int(s.checkm2_completeness.ge(50).sum()):,} at CheckM2
completeness ≥50% and median completeness {s.checkm2_completeness.median():.2f}%.
This is a different, non-exhaustive metadata category and should not be conflated
with the GORG BioProject membership screen.

Reproduction: `python scripts/249_audit_gorg_sag_source.py`. Exact selected records,
summary and input SHA256 hashes are stored alongside this report.
''')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
