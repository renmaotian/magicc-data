# Public scientific deposit

The current repository contains benchmark metadata, per-sample predictions,
scientific tables, source manifests and the minimal numerical/plotting/genus
training code identified in `reproduction/DEPENDENCIES.tsv`. The ordered
commands and tested scope are in `reproduction/README.md`.

The representative core-gene input archive is committed in two chunks under
`reproduction/inputs/core_gene_archive/`. `reproduction/ARCHIVES.json` binds the
combined 55,236,128-byte archive and all 2,003 member identities. It supplies the
actual 2,000 core-gene FASTAs used for feature reselection. The archive's container
metadata differ from an earlier prepared ZIP; member identities are preserved.

The 99,957 original genomic FASTAs are retrieved separately from NCBI using their
exact archived accession versions, compressed-file MD5 and original-file
SHA256. The committed source manifest covers 369,151,655,339 uncompressed bytes.
A bounded real download verifies the retrieval mechanism; it does not imply
that every historical accession will remain available indefinitely.

Unchanged benchmark assemblies remain on release `v1.0.0`; production MAGICC
and its frozen model remain on the software repository's `v0.3.3` release.
Research ONNX models are supplied separately with the submission materials and
are not uploaded to this data repository. Their full training and evaluation
recipe is public; no new GitHub release or research-model download is promised.

CAMI, Meslier/Zymo and catalogue raw sequences and comparator databases are
obtained from their original providers. Deposited predictions permit numerical
replay without these downloads. Full neural retraining needs the documented
large reference inputs and many CPU-hours; it is a separate workflow.

Manuscripts, reviewer correspondence, journal forms, author attestations and
private submission controls are excluded. Journal document rendering is not
part of this public scientific reproduction scope.
