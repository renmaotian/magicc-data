#!/usr/bin/env python3
"""Auditably change pooled observational novelty resampling to shared references."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5';WORK=OUT/'replay'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    src=ROOT/'scripts/210_ws11n_novelty_ladder.py';dst=WORK/'scripts/210_ws11n_novelty_ladder.py'
    s=src.read_text().replace(str(ROOT),str(WORK))
    s=s.replace('n_clusters_set_x_ref','n_clusters_shared_accession')
    old='D["cluster_id"] = D["set"] + "::" + D["dominant_accession"].astype(str)';assert old in s
    s=s.replace(old,'D["cluster_id"] = D["dominant_accession"].astype(str)')
    s=s.replace('if D["cluster_id"].nunique() != 2586:','if D["cluster_id"].nunique() != 1648:').replace('!= 2,586','!= 1,648')
    s=s.replace('set x dominant_accession','dominant_accession shared across sets').replace('set × dominant accession','dominant accession shared across sets')
    s=s.replace('set x reference genome (the WS5 definition)','shared dominant accessions across the five sets (resubmission5 correction)')
    s=s.replace('set x reference "','shared-reference "')
    s=s.replace('c: {"clusters": int(bal_cluster[c].sum()),','c: {"clusters": int(D.loc[D.novelty_class.eq(c), "cluster_id"].nunique()),')
    s=s.replace('if 0 < int(bal_cluster[c].sum()) < MIN_CLUSTERS_FOR_CLAIM],','if 0 < int(D.loc[D.novelty_class.eq(c), "cluster_id"].nunique()) < MIN_CLUSTERS_FOR_CLAIM],')
    s=s.replace('five leakage-free benchmark sets','five test-reference benchmark sets')
    s=s.replace('        "design": ("OBSERVATIONAL stratification', '        "pooled_weighting": "Each sample has equal weight within the novelty stratum; the same accession has one shared bootstrap multiplicity across sets.",\n        "historical_preprocessing_caveat": "Production V5 inherited unsupervised normalization fitted on V4 training, validation and test feature rows; held-out dominant accessions do not imply independent normalization.",\n        "design": ("OBSERVATIONAL stratification')
    dst.write_text(s)
    (OUT/'novelty_replay_change.json').write_text(json.dumps(dict(status='PREPARED',original_script=str(src.relative_to(ROOT)),original_sha256=sha(src),replay_script=str(dst.relative_to(ROOT)),replay_sha256=sha(dst),changes=['Official stage-corrected prediction inputs from replay','Shared dominant accession resampling across sets, sample-weighted strata','Distinct accession cluster-mean paired and trend analyses','Explicit preprocessing scope; no new predictions or changed novelty classification']),indent=2)+'\n')
    print('Prepared shared-reference novelty replay')
if __name__=='__main__':main()
