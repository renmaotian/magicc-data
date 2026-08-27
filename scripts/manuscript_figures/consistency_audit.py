#!/usr/bin/env python3
"""Cross-document numeric consistency sweep for the resubmission3 package.

Extracts every numeric quantity from ``manuscript_revised.md``,
``supplementary_revised.md`` and ``response_to_reviewers.md`` together with
enough surrounding context to say *what it measures*, then reports every
quantity that appears with two or more different values across (or within)
those files.

Design, and why it is not a bare-number diff
--------------------------------------------
Grouping by bare number would report thousands of coincidences ("5.31 appears
in three places").  Instead every number is annotated with

* a **unit class** (pp / % / seconds / GB / ratio / p-value / q-value / count …),
* a **label** (``MAE``, ``bias``, ``q =``, ``p =``, ``slope``, ``R2`` …) taken
  from the text immediately before it, and
* a **concept set** — canonical tokens for tool names, benchmark set names,
  cohort names, taxon names, taxonomic ranks, contamination types, thread
  counts and so on, matched against a curated vocabulary over a context window
  that, inside a markdown table, carries the section heading, the column header
  and the row label forward.

Three detectors then run over those records:

``STRICT``
    identical (unit, label, concept-set) with two or more distinct values.

``ULP``
    the same concept set (or a shared rare concept) where two values agree to
    all but the last significant digit — the misrounding / mistyping signature
    (``5.21`` vs ``5.20``; ``0.0005`` vs ``0.0006``; ``2.6e-8`` vs ``2.7e-8``).

``NEAR``
    the same shared rare concept where two values differ by no more than
    ``NEAR_REL`` in relative terms without being equal.

Numeric proximity cannot see a *semantic* error — a value quoted for the wrong
cohort, or a ratio that does not follow from the table beside it.  Those are
covered by ``PROBES``: declarative, individually documented checks at the foot
of this file.  Each probe names the result file that adjudicates it.

Usage
-----
    /path/to/conda/bin/python \\
        nature_communications/resubmission3/scripts/consistency_audit.py

Writes ``consistency_audit.tsv``, ``consistency_audit_records.tsv`` and
``consistency_audit_report.md`` next to the documents.  Exits non-zero when any
finding is reported, so it can be used as an acceptance gate.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, os.pardir))

DOCS = [
    "manuscript_revised.md",
    "supplementary_revised.md",
    "response_to_reviewers.md",
]

NEAR_REL = 0.06          # relative difference below which two values are "near"
MIN_CONCEPTS = 1         # a record needs at least this many concepts to group
RARE_MAX_SHARE = 0.16    # a concept is "rare" if it tags <= this share of records

# --------------------------------------------------------------------------
# number scanning
# --------------------------------------------------------------------------

SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
       "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+"}

#: A number, optionally signed, optionally with thousands separators, optionally
#: followed by a scientific-notation tail in either ``1.2e-3`` or
#: ``1.2 × 10⁻³`` form.
NUM_RE = re.compile(
    r"(?P<sign>[+\-−–])?"
    r"(?P<mant>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<exp>"
    r"\s*[eE][+\-−]?\d+"
    r"|\s*[×x]\s*10\s*[⁰¹²³⁴-⁻]+"
    r"|\s*[×x]\s*10\^[+\-−]?\d+"
    r")?"
)

#: Suffixes that fix the unit class, longest first.
UNITS = [
    ("pp", r"\s*(?:pp\b|percentage points?\b)"),
    ("pct", r"\s*%"),
    ("ratio", r"\s*[×x](?![\s]*10)"),
    ("gb", r"\s*GB\b"),
    ("mb", r"\s*MB\b"),
    ("mbp", r"\s*Mbp\b"),
    ("kb", r"\s*kbp?\b"),
    ("bp", r"\s*bp\b"),
    ("hour", r"\s*(?:h\b|hours?\b)"),
    ("min", r"\s*(?:m\b|min\b|minutes?\b)"),
    ("sec", r"\s*(?:s\b|seconds?\b)"),
]

#: Labels searched for in the 80 characters preceding a number; the closest
#: match wins.  Order is longest-first so that ``contamination MAE`` beats
#: ``MAE``.
LABELS = [
    ("q", r"\bq\s*(?:=|<|≤)\s*$"),
    ("p", r"\bp\s*(?:=|<|≤)\s*$"),
    ("n", r"\bn\s*=\s*$"),
    ("rho", r"(?:ρ|rho)\s*(?:=|of)\s*$"),
    ("phi", r"(?:φ|phi)\s*=\s*$"),
    ("delta", r"(?:δ|Cliff's delta)\s*(?:=)?\s*$"),
    ("r2", r"(?:R²|R\^?2|coefficient of determination|coefficients of determination)[^.;]{0,40}$"),
    ("comp_mae", r"completeness\s+(?:mean absolute error|MAE)[^.;|]{0,40}$"),
    ("cont_mae", r"contamination\s+(?:mean absolute error|MAE)[^.;|]{0,40}$"),
    ("mae", r"\bMAEs?\b[^.;|]{0,40}$"),
    ("bias", r"\bbias(?:es)?\b[^.;|]{0,40}$"),
    ("signed_error", r"signed errors?\b[^.;|]{0,40}$"),
    ("slope", r"\bslopes?\b[^.;|]{0,40}$"),
    ("macro_f1", r"macro\s*F1\b[^.;|]{0,30}$"),
    ("false_fail", r"false[- ]fail(?:\s+rate)?s?\b[^.;|]{0,30}$"),
    ("false_pass", r"false[- ]pass(?:\s+rate)?s?\b[^.;|]{0,30}$"),
    ("bal_acc", r"balanced accuracy\b[^.;|]{0,30}$"),
    ("did", r"(?:difference-in-differences|DiD)\b[^.;|]{0,40}$"),
    ("attenuation", r"attenuat\w+\b[^.;|]{0,30}$"),
    ("hl", r"(?:Hodges[–\-]Lehmann)\b[^.;|]{0,40}$"),
    ("rss", r"(?:peak[- ]RSS|peak resident set size|resident set size)\b[^.;|]{0,40}$"),
    ("wall", r"(?:wall[- ]clock|wall clock|takes|elapsed)\b[^.;|]{0,30}$"),
    ("footprint", r"(?:install footprint|footprint|reference data)\b[^.;|]{0,30}$"),
    ("elasticity", r"elasticity\b[^.;|]{0,30}$"),
]
LABELS_C = [(name, re.compile(rx)) for name, rx in LABELS]

# --------------------------------------------------------------------------
# concept vocabulary
# --------------------------------------------------------------------------

def _v(token: str, *patterns: str) -> "tuple[str, re.Pattern]":
    return token, re.compile("|".join(patterns), re.I)


VOCAB = [
    # tools ----------------------------------------------------------------
    _v("MAGICC", r"\bMAGICC\b"),
    _v("CHECKM2", r"\bCheckM2\b"),
    _v("COCOPYE", r"\bCoCoPyE\b"),
    _v("DEEPCHECK", r"\bDeepCheck\b"),
    _v("GUNC", r"\bGUNC\b"),
    _v("KRAKEN2", r"\bKraken\s*2\b"),
    _v("CHECKM1", r"\bCheckM\b(?!2)"),
    # model versions -------------------------------------------------------
    _v("V5", r"\bV5\b|\bmodel V5\b|magicc_v5"),
    _v("V4", r"\bV4\b|magicc_v4"),
    _v("V3", r"\bV3\b|magicc_v3"),
    # benchmark sets -------------------------------------------------------
    _v("SET_A", r"\bSet A\b|\bset_A\b"),
    _v("SET_B", r"\bSet B\b|\bset_B\b"),
    _v("SET_C_CLEAN", r"\bSets? C-clean\b|\bset_C_clean\b|C-clean"),
    _v("SET_D_CLEAN", r"\bSets? D-clean\b|\bset_D_clean\b|D-clean"),
    _v("SET_E", r"\bSet E\b|\bset_E\b"),
    _v("SET_F", r"\bSet F\b|\bset_F\b"),
    _v("SET_G", r"\bSet G\b|\bset_G\b"),
    _v("SET_H", r"\bSet H\b|\bset_H\b"),
    _v("FIVE_SET", r"five[- ]set|five leakage-free|pooled over 5,000|headline panel|Table S2[a-f]\b"),
    _v("LEAKY_PANEL", r"leaky panel|superseded leak|withdrawn panel|Table S2g\b"),
    _v("MOTIVATING", r"\bmotivating\b|motivating_v2|Table S1[a-d]\b"),
    # cohorts / datasets ---------------------------------------------------
    _v("MINION", r"\bMinION\b|\bminion\b|\bnanopore\b"),
    _v("PACBIO", r"\bPacBio\b|\bpacbio\b"),
    _v("ILLUMINA", r"\bIllumina\b"),
    _v("ION", r"\bIon\b|\bIon S5\b|\bIon Proton\b|\bproton\b|\bs5\b"),
    _v("MGISEQ", r"\bMGISEQ\b|mgiseq"),
    _v("ORF_INTACT", r"ORF[- ]intact"),
    _v("ORF_BROKEN", r"ORF[- ]compromised|ORF[- ]destruction|indel-dense"),
    _v("BALANCED_PANEL", r"balanced panel|balanced_panel|37 organisms|balanced 37"),
    _v("MESLIER", r"\bMeslier\b|\bMOCK1\b|mock[- ]community|mock community"),
    _v("ZYMO", r"\bZymoBIOMICS\b|\bZymo\b"),
    _v("CAMI", r"\bCAMI\s*II\b|\bCAMI\b|\bcami2\b"),
    _v("MARINE", r"\bmarine\b"),
    _v("STRAINMAD", r"strain[- ]madness"),
    _v("SPIRE", r"\bSPIRE\b"),
    _v("UHGG", r"\bUHGG\b"),
    _v("GTDB", r"\bGTDB\b"),
    _v("SAG", r"\bSAGs?\b|single[- ]cell amplified"),
    _v("NCBI", r"\bNCBI\b"),
    # taxa -----------------------------------------------------------------
    _v("PATESCI", r"Patescibacteri\w*|\bCPR\b"),
    _v("DPANN", r"\bDPANN\b"),
    _v("BACTEROIDOTA_A", r"Bacteroidota_A"),
    _v("BACTEROIDOTA", r"\bBacteroidota\b(?!_A)"),
    _v("HALOBACTERIOTA", r"Halobacteriota|halophilic"),
    _v("CAMPYLOBACTEROTA", r"Campylobacterota"),
    _v("HELICOBACTERACEAE", r"Helicobacteraceae"),
    _v("ARCOBACTERACEAE", r"Arcobacteraceae"),
    _v("MURIBACULACEAE", r"Muribaculaceae"),
    _v("FLAVOBACTERIACEAE", r"Flavobacteriaceae"),
    _v("HALOFAMILIES", r"Haloferacaceae|Haloarculaceae"),
    _v("SALMONELLA", r"Salmonella|Listeria"),
    # experiment axes ------------------------------------------------------
    _v("HOLDOUT_PHYLUM", r"leave-phylum-out|phylum[- ]holdout|phylum-level holdout|unseen phyl"),
    _v("HOLDOUT_FAMILY", r"leave-family-out|family[- ]holdout|family-level|unseen family|held-out famil"),
    _v("HOLDOUT_GENUS", r"leave-genus-out|genus[- ]holdout"),
    _v("DID", r"difference-in-differences|\bDiD\b"),
    _v("ATTENUATION", r"attenuat\w+"),
    _v("CONTROL", r"in[- ]distribution control|`?in_distribution`?"),
    _v("RANK_SPECIES", r"\bspecies\b"),
    _v("RANK_GENUS", r"\bgenus\b|\bgenera\b"),
    _v("RANK_FAMILY", r"\bfamily\b|\bfamilies\b"),
    _v("RANK_ORDER", r"\border\b(?!ing)"),
    _v("RANK_CLASS", r"\bclass\b(?!ification|ify)"),
    _v("RANK_PHYLUM", r"\bphylum\b|\bphyla\b"),
    _v("TYPE_REDUNDANT", r"\bredundant\b"),
    _v("TYPE_REPLACED", r"\breplaced\b"),
    _v("TYPE_SINGLE", r"\bsingle\b(?![- ]copy|[- ]cell)"),
    # error-robustness arms ------------------------------------------------
    _v("SUBSTITUTION", r"substitutions?\b|per[- ]base accuracy"),
    _v("INDEL", r"\bindels?\b"),
    _v("CHIMERA", r"chimeric|mis[- ]join"),
    _v("DUPLICATION", r"duplicat\w+|uneven[- ]coverage"),
    # metrics / quantities -------------------------------------------------
    _v("COMPLETENESS", r"completeness"),
    _v("CONTAMINATION", r"contaminat\w+"),
    _v("MACRO_F1", r"macro\s*F1"),
    _v("FALSE_FAIL", r"false[- ]fail"),
    _v("FALSE_PASS", r"false[- ]pass"),
    _v("BAL_ACC", r"balanced accuracy"),
    _v("MIMAG", r"MIMAG"),
    _v("SLOPE", r"\bslopes?\b|detection slope"),
    _v("ELASTICITY", r"elasticity|\bφ\b"),
    _v("LEAKAGE", r"leak\w+"),
    _v("RECALIBRATION", r"recalibrat\w+"),
    _v("CIRCULARITY", r"circularity|would[- ]have[- ](?:passed|failed)|would[- ](?:pass|fail)"),
    # performance ----------------------------------------------------------
    _v("WALLCLOCK", r"wall[- ]?clock|elapsed real time"),
    _v("RSS", r"peak[- ]RSS|resident set size|peak memory"),
    _v("THREADS1", r"\b1 thread\b|one thread\b|\|\s*1\s*\|"),
    _v("THREADS8", r"\b8 threads?\b"),
    _v("THREADS16", r"\b16 threads?\b"),
    _v("THREADS32", r"\b32 threads?\b"),
    _v("N100", r"100[- ]genome|\b100 genomes\b|0\.478 Gbp"),
    _v("N1000", r"1,000[- ]genome|\b1,000 genomes\b|4\.807 Gbp|4,806,980,767"),
    _v("SCALING", r"speed[- ]?up|parallel efficiency|scaling"),
    _v("TCO", r"total cost of ownership|install footprint|reference data|download"),
    _v("DOCKER", r"\bDocker\b|Apptainer|container"),
    # software versions ----------------------------------------------------
    _v("VER030", r"v?0\.3\.0"),
    _v("VER031", r"v?0\.3\.1"),
]

#: Two records may only be compared when, for every family below, either one of
#: them names no member of the family or the two sets of members overlap.  A
#: Set F number is never the same quantity as a Set G number; a MinION cell is
#: never an Ion cell; a motivating-panel row is never a leakage-free-panel row.
DISCRIM_FAMILIES = [
    {"SET_A", "SET_B", "SET_C_CLEAN", "SET_D_CLEAN", "SET_E", "SET_F",
     "SET_G", "SET_H"},
    {"MOTIVATING", "FIVE_SET", "LEAKY_PANEL"},
    {"MINION", "PACBIO", "ILLUMINA", "ION", "MGISEQ"},
    {"ORF_INTACT", "ORF_BROKEN"},
    {"MARINE", "STRAINMAD"},
    {"SUBSTITUTION", "INDEL", "CHIMERA", "DUPLICATION"},
    {"TYPE_REDUNDANT", "TYPE_REPLACED", "TYPE_SINGLE"},
    {"RANK_SPECIES", "RANK_GENUS", "RANK_FAMILY", "RANK_ORDER", "RANK_CLASS",
     "RANK_PHYLUM"},
    {"N100", "N1000"},
    {"THREADS1", "THREADS8", "THREADS16", "THREADS32"},
    {"V3", "V4", "V5"},
    {"VER030", "VER031"},
    {"PATESCI", "DPANN", "BACTEROIDOTA", "BACTEROIDOTA_A", "HALOBACTERIOTA",
     "CAMPYLOBACTEROTA", "HELICOBACTERACEAE", "ARCOBACTERACEAE",
     "MURIBACULACEAE", "FLAVOBACTERIACEAE", "HALOFAMILIES"},
    {"MAGICC", "CHECKM2", "COCOPYE", "DEEPCHECK", "GUNC"},
    {"SPIRE", "UHGG", "GTDB"},
    {"MESLIER", "ZYMO", "CAMI"},
    {"COMPLETENESS", "CONTAMINATION"},
]


def discriminators_conflict(a: "Record", b: "Record") -> bool:
    for fam in DISCRIM_FAMILIES:
        ia, ib = a.concepts & fam, b.concepts & fam
        if ia and ib and not (ia & ib):
            return True
    return False


#: Concepts too generic to key a group on their own.
BROAD = {
    "MAGICC", "CHECKM2", "COCOPYE", "DEEPCHECK", "COMPLETENESS", "CONTAMINATION",
    "V5", "GTDB", "NCBI", "LEAKAGE", "RANK_SPECIES", "RANK_GENUS", "RANK_FAMILY",
    "RANK_ORDER", "RANK_CLASS", "RANK_PHYLUM", "TYPE_SINGLE",
}


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

class Record:
    __slots__ = ("doc", "line", "raw", "value", "unit", "label", "concepts",
                 "in_ci", "context", "refs", "decimals", "in_table",
                 "ci", "table_id")

    ci: str
    table_id: str

    def __init__(self, doc, line, raw, value, unit, label, concepts, in_ci,
                 context, refs, decimals=0, in_table=False, ci="",
                 table_id=""):
        self.ci = ci
        self.table_id = table_id
        self.doc = doc
        self.line = line
        self.raw = raw
        self.value = value
        self.unit = unit
        self.label = label
        self.concepts = concepts
        self.in_ci = in_ci
        self.context = context
        self.refs = refs
        self.decimals = decimals
        self.in_table = in_table

    def where(self) -> str:
        return f"{self.doc}:{self.line}"

    def is_measurement(self) -> bool:
        """Filter out counts, years, citation numbers and identifiers.

        A quantity that can be *misrounded* between two documents always
        carries a decimal fraction, so requiring one removes almost all of the
        noise (reference call-outs ``(10,17,18)``, years, sample counts,
        thread counts) at no cost to the class of defect this sweep hunts.
        """
        if self.decimals < 1 or self.in_ci:
            return False
        if 1900 <= self.value <= 2100 and self.decimals == 0:
            return False
        if len(self.concepts) > 18:      # kitchen-sink context, too vague
            return False
        return bool(self.unit or self.label or self.in_table)


def _to_float(m: "re.Match") -> "float | None":
    mant = m.group("mant").replace(",", "")
    try:
        val = float(mant)
    except ValueError:
        return None
    sign = m.group("sign")
    if sign in ("-", "−", "–"):
        val = -val
    exp = m.group("exp")
    if exp:
        e = "".join(SUP.get(ch, ch) for ch in exp)
        e = e.replace("×", "").replace("x", "").replace("^", "")
        e = re.sub(r"\s+", "", e)
        if e.lower().startswith("e"):
            e = e[1:]
        elif e.startswith("10"):
            e = e[2:]
        e = e.replace("−", "-")
        if e in ("", "+", "-"):
            return val
        try:
            val *= 10.0 ** int(e)
        except ValueError:
            return val
    return val


def _unit_of(line: str, end: int) -> str:
    tail = line[end:end + 20]
    for name, rx in UNITS:
        if re.match(rx, tail):
            return name
    return ""


def _label_of(line: str, start: int) -> str:
    head = line[max(0, start - 80):start]
    best, best_pos = "", -1
    for name, rx in LABELS_C:
        m = rx.search(head)
        if m and m.start() > best_pos:
            best, best_pos = name, m.start()
    return best


def _concepts_of(text: str) -> "frozenset[str]":
    return frozenset(tok for tok, rx in VOCAB if rx.search(text))


REF_RE = re.compile(
    r"(?:Table|Tables|Fig\.|Figs\.|Figure|Figures|Supplementary Table|"
    r"Supplementary Figure|Supplementary Note)\s*S?\d+[a-z]?")


def _refs_of(text: str) -> str:
    return ",".join(sorted(set(REF_RE.findall(text))))


HEAD_RE = re.compile(r"^(#{1,6})\s+(.*)$")
ROW_RE = re.compile(r"^\s*\|(.*)\|\s*$")
SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
SENT_SPLIT = re.compile(r"(?<=[.;:])\s+")


def scan(path: str, doc: str) -> "list[Record]":
    """Return one Record per number found in *path*."""
    lines = open(path, encoding="utf-8").read().split("\n")
    out: "list[Record]" = []

    section = ""       # nearest '# ' heading
    heading = ""       # nearest deeper heading (## / ###)
    header_cells: "list[str]" = []
    row_label = ""
    table_id = ""
    in_table = False
    prev_row: "list[str] | None" = None
    para_prev = ""     # previous non-empty prose line, for sentence fallback

    for i, line in enumerate(lines, start=1):
        hm = HEAD_RE.match(line)
        if hm:
            if len(hm.group(1)) == 1:
                section, heading = hm.group(2), ""
            else:
                heading = hm.group(2)
            in_table, header_cells, row_label, prev_row = False, [], "", None
            table_id = ""
            if any(ch.isdigit() for ch in hm.group(2)):
                ctx = " || ".join(x for x in (section, heading) if x)
                _emit(out, doc, i, hm.group(2), ctx or hm.group(2),
                      _concepts_of(ctx or hm.group(2)),
                      _refs_of(ctx or hm.group(2)), table=False)
            continue

        rm = ROW_RE.match(line)
        if rm and SEP_RE.match(line):
            # separator: the row before it was the header
            if prev_row is not None:
                header_cells = prev_row
                in_table = True
                table_id = f"{doc}#t{i}"
            continue
        if rm:
            cells = [c.strip() for c in rm.group(1).split("|")]
            prev_row = cells
            if in_table:
                if cells and cells[0]:
                    row_label = cells[0]
                _scan_table_row(out, doc, i, line, cells, header_cells,
                                row_label, section, heading, table_id)
                continue
            # header row not yet confirmed; fall through and treat as prose too
        else:
            prev_row = None
            if line.strip():
                in_table = False

        if not line.strip():
            continue
        _scan_prose_line(out, doc, i, line, section, heading, para_prev)
        para_prev = line

    return out


def _scan_table_row(out, doc, lineno, line, cells, header_cells, row_label,
                    section, heading, table_id=""):
    for idx, cell in enumerate(cells):
        if not cell:
            continue
        col = header_cells[idx] if idx < len(header_cells) else ""
        if re.search(r"range|min\s*[–\-]\s*max|\bmin\b|\bmax\b",
                     col, re.I):
            continue          # a range column holds no point estimate
        ctx = " || ".join(x for x in (section, heading, row_label, col, cell) if x)
        concepts = _concepts_of(ctx)
        refs = _refs_of(ctx)
        _emit(out, doc, lineno, cell, ctx, concepts, refs, table=True,
              table_id=table_id)


def _scan_prose_line(out, doc, lineno, line, section, heading, para_prev):
    parts = SENT_SPLIT.split(line)
    pos = 0
    for sent in parts:
        idx = line.find(sent, pos)
        if idx < 0:
            idx = pos
        pos = idx + len(sent)
        ctx = " || ".join(x for x in (section, heading, sent) if x)
        concepts = _concepts_of(ctx)
        if len(concepts) < 2:
            # a short clause inherits its neighbours' context
            wide = " || ".join(x for x in (section, heading, para_prev, line) if x)
            concepts = _concepts_of(wide)
            ctx_refs = wide
        else:
            ctx_refs = ctx
        refs = _refs_of(ctx_refs)
        _emit(out, doc, lineno, sent, ctx, concepts, refs, table=False,
              base=line, offset=idx)


#: Display-item call-outs and bare reference-citation groups, whose digits are
#: identifiers rather than measurements.
IDENT_BEFORE = re.compile(
    r"(?:Tables?|Figs?\.?|Figures?|Notes?|Supplementary (?:Table|Figure|Note)|"
    r"Set|set_[A-H]_?\w*|v|V|r|R|opset|IR|SHA256|GCA_|GCF_|PRJEB|§)\s*S?$")
CITE_GROUP = re.compile(r"\((\d{1,2}(?:\s*[,–\-]\s*\d{1,2})*)\)")


#: ``4.62 [4.32, 4.92]`` / ``+5.21 pp [4.55–5.79]`` — a point estimate and the
#: interval that belongs to it.  The interval travels with the point estimate so
#: that a lone bracket bound is never compared against an unrelated one.
PT_CI = re.compile(
    r"(?P<pt>[+\-−–]?\d[\d,]*(?:\.\d+)?)\s*(?:pp|%|×|s|GB)?\s*"
    r"\[\s*(?P<lo>[+\-−–]?[\d.,]+)\s*(?:,|–|—|-|to)\s*(?P<hi>[+\-−–]?[\d.,]+)\s*\]")


def _norm_ci(lo: str, hi: str) -> str:
    def n(x):
        x = x.replace(",", "").replace("−", "-").replace("–", "-").strip()
        try:
            return f"{float(x):g}"
        except ValueError:
            return x
    return f"[{n(lo)}, {n(hi)}]"


def _emit(out, doc, lineno, text, ctx, concepts, refs, table, base=None,
          offset=0, table_id=""):
    cite_spans = [(m.start(), m.end()) for m in CITE_GROUP.finditer(text)]
    ci_at = {}
    for m in PT_CI.finditer(text):
        ci_at[m.start("pt")] = _norm_ci(m.group("lo"), m.group("hi"))
    for m in NUM_RE.finditer(text):
        start = m.start()
        look = text[max(0, start - 14):start]
        if IDENT_BEFORE.search(look):
            continue
        if any(a <= start < b for a, b in cite_spans):
            continue
        val = _to_float(m)
        if val is None:
            continue
        unit = _unit_of(text, m.end())
        label = _label_of(text, start)
        in_ci = _inside_brackets(text, start)
        raw = text[start:m.end() + (3 if unit else 0)].strip()
        out.append(Record(doc, lineno, raw, val, unit, label, concepts, in_ci,
                          _short(ctx), refs, _decimals(m.group("mant")), table,
                          ci_at.get(start, ""), table_id))


def _inside_brackets(text: str, pos: int) -> bool:
    op = text.rfind("[", 0, pos)
    if op < 0:
        return False
    cl = text.find("]", op)
    return cl == -1 or cl > pos


def _short(ctx: str, n: int = 240) -> str:
    ctx = re.sub(r"\s+", " ", ctx).strip()
    return ctx if len(ctx) <= n else ctx[:n] + "…"


# --------------------------------------------------------------------------
# comparison helpers
# --------------------------------------------------------------------------

def _decimals(raw: str) -> int:
    m = re.search(r"\.(\d+)", raw.replace(",", ""))
    return len(m.group(1)) if m else 0


def _sigfig_parts(v: float) -> "tuple[float, int]":
    """Return (mantissa, exponent) with the mantissa in [1, 10)."""
    if v == 0:
        return 0.0, 0
    import math
    e = math.floor(math.log10(abs(v)))
    return v / (10.0 ** e), e


def one_ulp_apart(a: Record, b: Record) -> bool:
    """True when two values agree except in the last printed significant digit."""
    if a.value == b.value:
        return False
    da, db = _decimals(a.raw), _decimals(b.raw)
    if da and da == db:
        scale = 10.0 ** da
        return abs(round(a.value * scale) - round(b.value * scale)) == 1
    # scientific notation: compare mantissas at equal precision
    ma, ea = _sigfig_parts(a.value)
    mb, eb = _sigfig_parts(b.value)
    if ea != eb:
        return False
    if da and db and da != db:
        return False
    prec = max(da, db, 1)
    scale = 10.0 ** prec
    return abs(round(ma * scale) - round(mb * scale)) == 1


def near(a: Record, b: Record) -> bool:
    """Cross-document only: two documents giving nearly the same number for
    what the context says is the same quantity is worth a human look; the same
    thing inside one document is nearly always two adjacent table cells."""
    if a.value == b.value or a.doc == b.doc:
        return False
    denom = max(abs(a.value), abs(b.value))
    if denom == 0:
        return False
    return abs(a.value - b.value) / denom <= NEAR_REL


def ci_mismatch(a: Record, b: Record) -> bool:
    """Same point estimate, different confidence interval."""
    if not a.ci or not b.ci:
        return False
    return a.value == b.value and a.ci != b.ci


# --------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------

class Finding:
    def __init__(self, kind, key, records, note=""):
        self.kind = kind
        self.key = key
        self.records = records
        self.note = note

    def values(self):
        seen, out = set(), []
        for r in self.records:
            if r.value not in seen:
                seen.add(r.value)
                out.append(r)
        return out


def detect_strict(records) -> "list[Finding]":
    """Identical (unit, label, concept-set) carrying two or more values."""
    groups = defaultdict(list)
    for r in records:
        if not r.is_measurement() or not r.label:
            continue
        if len(r.concepts) < MIN_CONCEPTS:
            continue
        groups[(r.unit, r.label, r.in_ci, r.concepts)].append(r)
    out = []
    for key, rs in groups.items():
        vals = {r.value for r in rs}
        if len(vals) < 2:
            continue
        if len({r.doc for r in rs}) < 2:
            continue                       # within one document, usually a table
        unit, label, in_ci, concepts = key
        out.append(Finding(
            "STRICT",
            f"{label}|{unit or '-'}|{'CI' if in_ci else 'pt'}|"
            f"{'+'.join(sorted(concepts))}",
            sorted(rs, key=lambda r: (r.doc, r.line))))
    return out


def _rare_concepts(records) -> "set[str]":
    counts = defaultdict(int)
    for r in records:
        for c in r.concepts:
            counts[c] += 1
    n = max(1, len(records))
    return {c for c, k in counts.items()
            if c not in BROAD and k / n <= RARE_MAX_SHARE}


def _units_compatible(a: Record, b: Record) -> bool:
    """A table cell usually carries no unit; its column header does."""
    if a.unit and b.unit:
        return a.unit == b.unit
    return True


def detect_pairs(records, kind, predicate) -> "list[Finding]":
    """Pair up measurement records that share a rare concept.

    One finding per pair, so every row of the report is a single actionable
    disagreement rather than a bucket.
    """
    meas = [r for r in records if r.is_measurement()]
    rare = _rare_concepts(meas)
    buckets = defaultdict(list)
    for r in meas:
        for c in r.concepts & rare:
            buckets[c].append(r)

    seen_pairs, out = set(), []
    for concept, rs in buckets.items():
        if len(rs) < 2 or len(rs) > 1600:
            continue
        for i in range(len(rs)):
            a = rs[i]
            for j in range(i + 1, len(rs)):
                b = rs[j]
                if a.doc == b.doc and a.line == b.line:
                    continue
                # two cells of one table are two different quantities by
                # construction, so never compare across rows of the same table
                if a.table_id and a.table_id == b.table_id:
                    continue
                if a.label and b.label and a.label != b.label:
                    continue
                if not _units_compatible(a, b):
                    continue
                if discriminators_conflict(a, b):
                    continue
                shared = a.concepts & b.concepts
                if len(shared - BROAD) < 2 and len(shared) < 3:
                    continue
                if not predicate(a, b):
                    continue
                sig = tuple(sorted([(a.doc, a.line, a.raw),
                                    (b.doc, b.line, b.raw)]))
                if sig in seen_pairs:
                    continue
                seen_pairs.add(sig)
                label = a.label or b.label or "-"
                unit = a.unit or b.unit or "-"
                out.append(Finding(
                    kind,
                    f"{label}|{unit}|{'CI' if a.in_ci else 'pt'}|"
                    f"{'+'.join(sorted(shared - BROAD)) or concept}",
                    sorted([a, b], key=lambda r: (r.doc, r.line))))
    return out


# --------------------------------------------------------------------------
# semantic probes — things numeric proximity structurally cannot see
# --------------------------------------------------------------------------

def probe_partition(texts) -> "list[Finding]":
    """"Of N pairwise comparisons, A survive …, B are losses and C are ties"."""
    out = []
    rx = re.compile(
        r"Of\s+(\d+)\s+pairwise comparisons?[^.]{0,200}?"
        r"(\w+|\d+)\s+survive[^.]{0,120}?correction,\s*"
        r"(\w+|\d+)\s+(?:are|is)\s+losses?[^.]{0,60}?"
        r"(\w+|\d+)\s+(?:are|is)\s+ties?", re.I)
    words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

    def num(tok):
        tok = tok.strip().lower()
        return int(tok) if tok.isdigit() else words.get(tok)

    for doc, lines in texts.items():
        for i, line in enumerate(lines, start=1):
            for m in rx.finditer(line):
                total, surv, loss, tie = (num(m.group(k)) for k in (1, 2, 3, 4))
                if None in (total, surv, loss, tie):
                    continue
                if surv + loss + tie != total:
                    r = Record(doc, i, m.group(0)[:160], float(total), "", "",
                               frozenset({"FIVE_SET"}), False,
                               _short(m.group(0)), "")
                    out.append(Finding(
                        "PROBE:partition",
                        "win/loss/tie partition does not sum",
                        [r],
                        f"stated {surv} survive + {loss} losses + {tie} ties = "
                        f"{surv + loss + tie}, against a stated total of {total}. "
                        "results/revision/metrics/ws5.4_clustered_tests.tsv gives "
                        "23 wins, 5 losses and 2 ties (28 significant under BH, "
                        "2 not)."))
    return out


def probe_table1_rss(texts) -> "list[Finding]":
    """A quoted peak-RSS ratio must name the cell it comes from."""
    out = []
    rx = re.compile(r"peak[- ]RSS ratio of\s*([\d.]+)\s*[×x]")
    for doc, lines in texts.items():
        for i, line in enumerate(lines, start=1):
            for m in rx.finditer(line):
                val = float(m.group(1))
                clause = line[m.start():m.end() + 140]
                names_cell = re.search(r"100[- ]genome|Table S4d|0\.478 Gbp",
                                       clause)
                window = line[max(0, m.start() - 200):m.end() + 200]
                if not names_cell:
                    r = Record(doc, i, m.group(0), val, "ratio", "rss",
                               frozenset({"RSS", "N100", "CHECKM2", "MAGICC"}),
                               False, _short(window), "Table 1,Table S4d")
                    out.append(Finding(
                        "PROBE:rss_ratio",
                        "peak-RSS ratio quoted without its cell",
                        [r],
                        f"{val}x is the 100-genome cell (Table S4d, "
                        "results/revision/speed/). Table 1 is the 1,000-genome "
                        "cell and implies 18.88 / 0.50 = 37.8x. The sentence must "
                        "name the cell."))
    return out


def probe_cohort_mixing(texts) -> "list[Finding]":
    """The Meslier MinION numbers must not cross cohorts inside one clause."""
    out = []
    panel = {"42.21", "6.75", "8.91"}     # fragmentation_gradient balanced_panel_all
    cohort = {"4.91", "50.69", "56.83", "3.11", "3.12"}  # metrics_by_cohort n = 39
    for doc, lines in texts.items():
        for i, line in enumerate(lines, start=1):
            for clause in re.split(r"(?<=[.])\s+", line):
                hits_p = {v for v in panel if v in clause}
                hits_c = {v for v in cohort if v in clause}
                if hits_p and hits_c:
                    r = Record(doc, i, ", ".join(sorted(hits_p | hits_c)), 0.0,
                               "pp", "comp_mae",
                               frozenset({"MINION", "MESLIER", "BALANCED_PANEL"}),
                               False, _short(clause), "Table S16a")
                    out.append(Finding(
                        "PROBE:cohort_mixing",
                        "balanced-panel and 39-bin MinION values in one clause",
                        [r],
                        "42.21 / 6.75 / 8.91 are the balanced_panel_all arm "
                        "(n = 49 bins) of "
                        "results/revision/real_data/meslier/fragmentation_gradient.tsv; "
                        "4.91 / 50.69 / 3.11 / 56.83 are the "
                        "ORF_compromised_assembly_only cohort (n = 39 bins) of "
                        "results/revision/real_data/meslier/metrics_by_cohort.tsv. "
                        "Split into two statements, each naming its cohort and n."))
    return out


def probe_minion_row(texts) -> "list[Finding]":
    """The n = 39 MinION row must read MAGICC 4.91 / CheckM2 50.69 / CoCoPyE 3.11 /
    DeepCheck 56.83 — a shifted value here is a scrambled row."""
    out = []
    rx = re.compile(
        r"CheckM2'?s?\s*([\d.]+)\s*pp[^.]{0,80}?CoCoPyE'?s?[^.]{0,40}?"
        r"DeepCheck'?s?\s*([\d.]+)\s*and\s*([\d.]+)\s*pp")
    for doc, lines in texts.items():
        for i, line in enumerate(lines, start=1):
            if "4.91" not in line:
                continue
            for m in rx.finditer(line):
                c2, co, dc = m.group(1), m.group(2), m.group(3)
                if (c2, co, dc) != ("50.69", "3.11", "56.83"):
                    r = Record(doc, i, m.group(0)[:160], float(c2), "pp",
                               "comp_mae",
                               frozenset({"MINION", "MESLIER", "ORF_BROKEN"}),
                               False, _short(line), "Table S16a")
                    out.append(Finding(
                        "PROBE:minion_row",
                        "MinION-only (n = 39) completeness MAE row is scrambled",
                        [r],
                        f"reads CheckM2 {c2}, CoCoPyE {co}, DeepCheck {dc}; "
                        "results/revision/real_data/meslier/metrics_by_cohort.tsv "
                        "cohort ORF_compromised_assembly_only gives MAGICC 4.9048, "
                        "CheckM2 50.6879, CoCoPyE 3.1145, DeepCheck 56.8308."))
    return out


def probe_version(texts) -> "list[Finding]":
    """v0.3.0 and v0.3.1 must not both describe the released package."""
    out = []
    hits = []
    for doc, lines in texts.items():
        for i, line in enumerate(lines, start=1):
            for m in re.finditer(r"v?0\.3\.[01]", line):
                hits.append((doc, i, m.group(0), _short(line)))
    vers = {h[2].lstrip("v") for h in hits}
    if len(vers) > 1:
        rs = [Record(d, i, raw, 0.3, "", "version",
                     frozenset({"VER030" if raw.endswith("0") else "VER031"}),
                     False, ctx, "") for d, i, raw, ctx in hits]
        out.append(Finding(
            "PROBE:version",
            "released software version quoted two ways",
            rs,
            "pyproject.toml is 0.3.1, the git tag is v0.3.1 and PyPI serves "
            "0.3.1; 0.3.0 lacks gzip input and --input-list, which the timing "
            "campaign and the reproduction workflow both exercise. Every "
            "non-historical mention must say v0.3.1."))
    return out


PROBES = [probe_partition, probe_table1_rss, probe_cohort_mixing,
          probe_minion_row, probe_version]


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def severity(f: Finding) -> int:
    """3 = certain defect, 2 = probable, 1 = review."""
    if f.kind.startswith("PROBE"):
        return 3
    docs = {r.doc for r in f.records}
    if f.kind == "ULP":
        return 3 if len(docs) > 1 else 2
    if f.kind == "CI":
        return 3 if len(docs) > 1 else 2
    if f.kind == "STRICT":
        return 2 if len(docs) > 1 else 1
    return 1


def render(findings, out_dir) -> int:
    findings = sorted(findings,
                      key=lambda f: (-severity(f), f.kind, f.key))
    tsv = os.path.join(out_dir, "consistency_audit.tsv")
    with open(tsv, "w", encoding="utf-8") as fh:
        fh.write("id\tseverity\tkind\tkey\tn_values\tvalues\tdocuments\t"
                 "locations\tnote\n")
        for n, f in enumerate(findings, start=1):
            vals = f.values()
            fh.write("\t".join([
                f"C{n:03d}",
                str(severity(f)),
                f.kind,
                f.key,
                str(len(vals)),
                " | ".join((r.raw + " " + r.ci).strip() for r in vals),
                ",".join(sorted({r.doc for r in f.records})),
                " ; ".join(f"{r.where()}={r.raw}" for r in f.records[:12]),
                f.note.replace("\t", " ").replace("\n", " "),
            ]) + "\n")

    rpt = os.path.join(out_dir, "consistency_audit_report.md")
    with open(rpt, "w", encoding="utf-8") as fh:
        fh.write("# Cross-document numeric consistency sweep\n\n")
        fh.write(f"`scripts/consistency_audit.py` over {', '.join(DOCS)}.\n\n")
        fh.write(f"**{len(findings)} findings.**  Severity 3 = certain defect, "
                 "2 = probable, 1 = review by eye.\n\n")
        by_kind = defaultdict(int)
        for f in findings:
            by_kind[f.kind] += 1
        fh.write("| kind | findings |\n|---|---|\n")
        for k in sorted(by_kind):
            fh.write(f"| {k} | {by_kind[k]} |\n")
        fh.write("\n---\n\n")
        for n, f in enumerate(findings, start=1):
            fh.write(f"## C{n:03d} · sev {severity(f)} · {f.kind} · `{f.key}`\n\n")
            if f.note:
                fh.write(f"> {f.note}\n\n")
            for r in f.records[:14]:
                fh.write(f"- `{r.where()}` **{r.raw}**"
                         f"{'  ' + r.ci if r.ci else ''}"
                         f"{'  refs: ' + r.refs if r.refs else ''}\n")
                fh.write(f"    - {r.context}\n")
            if len(f.records) > 14:
                fh.write(f"- … {len(f.records) - 14} more\n")
            fh.write("\n")
    return len(findings)


def dump_records(records, out_dir) -> None:
    path = os.path.join(out_dir, "consistency_audit_records.tsv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("doc\tline\traw\tvalue\tunit\tlabel\tci\tconcepts\trefs\t"
                 "context\n")
        for r in records:
            fh.write("\t".join([
                r.doc, str(r.line), r.raw, repr(r.value), r.unit, r.label,
                "1" if r.in_ci else "0", "+".join(sorted(r.concepts)), r.refs,
                r.context.replace("\t", " "),
            ]) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=PKG, help="package directory")
    ap.add_argument("--min-severity", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    records, texts = [], {}
    for doc in DOCS:
        path = os.path.join(args.dir, doc)
        if not os.path.exists(path):
            print(f"missing: {path}", file=sys.stderr)
            return 2
        records.extend(scan(path, doc))
        texts[doc] = open(path, encoding="utf-8").read().split("\n")

    findings = []
    findings += detect_strict(records)
    findings += detect_pairs(records, "ULP", one_ulp_apart)
    findings += detect_pairs(records, "CI", ci_mismatch)
    findings += detect_pairs(records, "NEAR", near)
    for probe in PROBES:
        findings += probe(texts)

    # ULP subsumes NEAR for the same pair of locations
    sharp = {(r.doc, r.line, r.raw) for f in findings
             if f.kind in ("ULP", "CI") for r in f.records}
    findings = [f for f in findings
                if f.kind != "NEAR"
                or not all((r.doc, r.line, r.raw) in sharp for r in f.records)]
    findings = [f for f in findings if severity(f) >= args.min_severity]

    dump_records(records, args.dir)
    n = render(findings, args.dir)

    if not args.quiet:
        print(f"records scanned : {len(records)}")
        print(f"findings        : {n}")
        for sev in (3, 2, 1):
            k = sum(1 for f in findings if severity(f) == sev)
            print(f"  severity {sev}    : {k}")
        print()
        print("wrote consistency_audit.tsv, consistency_audit_records.tsv, "
              "consistency_audit_report.md")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
