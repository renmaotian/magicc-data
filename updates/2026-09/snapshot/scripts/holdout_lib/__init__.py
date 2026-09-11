"""Frozen snapshot of magicc/ pipeline modules for the WS1.6/1.7/1.8
leave-phylum-out experiment (Nature Communications revision, Stage B).

fragmentation.py, contamination.py, assembly_stats.py, kmer_counter.py,
normalization.py, storage.py, model.py and trainer.py are copies of the
corresponding magicc/*.py files at git commit 5b6a9a6 ("Release V5 model:
bump version to 0.3.0"), i.e. exactly the code that produced production V5.

The ONLY edits are two import lines, changed so the package is self-contained
and cannot silently fall back on the shared, concurrently-edited magicc package:
    contamination.py:14  magicc.fragmentation  -> .fragmentation
    normalization.py:20  magicc.assembly_stats -> .assembly_stats
Both target modules are byte-identical copies, so behaviour is unchanged
(verify with `diff magicc/<m>.py scripts/holdout_lib/<m>.py`).

config.py and seeding.py are new and specific to this experiment.

Rationale for freezing: other agents are concurrently editing
magicc/fragmentation.py and magicc/cli.py. Freezing guarantees that the holdout
synthesis, training and evaluation use exactly the V5 code path even if the
shared modules change, and makes the experiment reproducible from this directory.

Provenance hashes: FROZEN_SHA256.txt (this package) and
FROZEN_SHA256_SOURCE.txt (the magicc/ files at copy time).
"""
