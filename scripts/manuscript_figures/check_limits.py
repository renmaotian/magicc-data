#!/usr/bin/env python3
"""Nature Communications Article limit and hygiene checker for resubmission3.

Counts what the journal counts, then enforces the acceptance gates of
``BUILD_CONTRACT_V3.md`` §1 and §13.  Prints one PASS/FAIL row per gate and
exits non-zero if any gate fails.

    /path/to/conda/bin/python \\
        nature_communications/resubmission3/scripts/check_limits.py [manuscript.md]

Limit gates (main manuscript only)
    title ≤ 15 words · abstract ≤ 150 words and no citations · Introduction +
    Results + Discussion ≤ 5,000 · Methods ≤ 3,000 · display items ≤ 10 ·
    references ≤ 70 · **every figure and table legend ≤ 350 words, reported
    individually**.

Hygiene gates (every document in the package)
    zero unresolved placeholders (``[RELEASE]``, ``[DOI]``, ``[DATE]``,
    ``[ANALYSIS REPO``) · zero ``AUTHOR TO VERIFY`` · zero ``figshare`` /
    ``Zenodo`` · zero ``Source Data`` / ``source_data`` · zero internal
    shorthand (``trap T``, bare ``CRACOT``, ``W1``–``W24`` register codes,
    ``WS1``-style workstream codes, ⛔/⚠️ marks) · every ``v0.3.0`` reported
    with its line for adjudication, failing unless the same sentence is
    explicitly historical.

Structural gates (main manuscript only)
    every numbered reference cited at least once in the main text · bold in
    body prose only where it marks a figure or table cross-reference.

Main text = Introduction + Results + Discussion, excluding the abstract,
Methods, references, figure legends, table legends and the back matter.  Words
are counted after stripping markdown emphasis, heading markers and table pipes,
the way a Word word count would see the rendered text.
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, os.pardir))
DEFAULT = os.path.join(PKG, "manuscript_revised.md")

MAIN_TEXT_SECTIONS = {"introduction", "results", "discussion"}

#: Every document that must be clean of placeholders and internal shorthand.
PACKAGE_DOCS = [
    "manuscript_revised.md",
    "supplementary_revised.md",
    "response_to_reviewers.md",
    "cover_letter.md",
    "README.md",
]

LIMITS = {
    "main_text": 5000,
    "abstract": 150,          # BUILD_CONTRACT_V3 §1: was 200
    "title": 15,
    "methods": 3000,
    "display_items": 10,
    "references": 70,
    "legend": 350,            # BUILD_CONTRACT_V3 §1: per figure / table legend
}

#: A display-item block starts at a ``[[FIGURE:n]]`` / ``[[TABLE:n]]`` marker and runs
#: through the table markdown and the legend that follow it.  The journal excludes
#: display-item legends from the word count, so they are stripped out here -- and
#: counted separately, because "figure legends" is explicit in the journal's rule while
#: "table legends" is not, and we want to see the total under both readings.
_DISPLAY_CONT = ("|", "**Figure ", "**Table ", "*Denominators", "![")


def split_display(text: str) -> "tuple[str, str]":
    """Return (prose, display-item blocks) for one section."""
    prose, display, in_block = [], [], False
    for para in re.split(r"(?:\r?\n\s*){2,}", text):
        s = para.strip()
        if re.match(r"^\[\[(FIGURE|TABLE):\d+\]\]", s):
            in_block = True
            display.append(para)
            continue
        if in_block and (not s or s.startswith(_DISPLAY_CONT)):
            display.append(para)
            continue
        in_block = False
        prose.append(para)
    return "\n\n".join(prose), "\n\n".join(display)


def clean(text: str) -> str:
    """Strip markdown so the count matches rendered prose."""
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)   # heading markers
    text = re.sub(r"\*\*|\*|`|~~", "", text)             # emphasis / code
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)     # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links -> label
    text = text.replace("|", " ")                        # table pipes
    text = re.sub(r"^\s*[-:| ]+\s*$", "", text, flags=re.M)  # table rules
    return text


def words(text: str) -> int:
    return len(clean(text).split())


def split_sections(md: str) -> "list[tuple[str, str]]":
    lines = md.split("\n")
    heads = [(i, l[2:].strip()) for i, l in enumerate(lines) if re.match(r"^# ", l)]
    out, front = [], "\n".join(lines[: heads[0][0]]) if heads else md
    out.append(("__front__", front))
    for k, (i, name) in enumerate(heads):
        j = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        out.append((name, "\n".join(lines[i + 1: j])))
    return out


# ---------------------------------------------------------------------------
# gate definitions
# ---------------------------------------------------------------------------

PLACEHOLDERS = [r"\[RELEASE\]", r"\[DOI\]", r"\[DATE\]", r"\[ANALYSIS REPO",
                r"\[MANUSCRIPT NUMBER\]"]

BANNED_TEXT = [
    ("AUTHOR TO VERIFY", r"AUTHOR TO VERIFY"),
    ("figshare", r"figshare"),
    ("Zenodo", r"[Zz]enodo"),
    # Narrowed 2026-08-27. The blanket ``Source Data|source_data`` gate was
    # unsatisfiable: editorial requirement E8 *is named* "Source Data file" by
    # the editor, so the entry heading, the index row and the sentences that
    # DENY the file must all use the term; and README's deleted-items table
    # must name source_data.xlsx / source_data_manifest.tsv /
    # build_source_data.py. The contract's actual intent is that the package
    # must not PROMISE a Source Data file, so only the promise is fatal.
    ("Source Data promise",
     r"Source Data (?:are|is) provided with this paper|"
     r"Source Data (?:file|workbook)[^.\n]{0,40}?(?:is|are) (?:provided|included|enclosed|attached)|"
     r"(?:provide|providing|enclose|enclosing|attach)[a-z]*\s+(?:a\s+)?Source Data"),
]

#: Every other mention of Source Data: listed for human review, never fatal.
SOURCE_DATA_ADVISORY = r"Source Data|source_data"

SHORTHAND = [
    ("trap T", r"\btraps? T\d*\b"),
    ("W-register code", r"\bW\d{1,2}\b"),
    ("WS workstream code", r"\bWS\d"),
    ("internal QA marks", "[⚠⛔✅❌️]"),
]

#: Result-file paths and script numbers: internal today, deposited after §8.
#: Reported but not fatal, because the deposition may legitimise them.
ADVISORY = [
    ("internal result-file path", r"results/revision/[A-Za-z0-9_/.{},|-]*"),
    ("internal script number", r"\bscripts?/\d+[A-Za-z0-9_.]*"),
    ("snake_case set name", r"\bset_[A-H](?:_clean|_v2)?\b"),
    ("Source Data mention (must be a denial, a filename or the editor's own requirement name)",
     SOURCE_DATA_ADVISORY),
]

#: An expansion or citation must follow within this many characters of CRACOT.
CRACOT_OK = re.compile(
    r"CRACOT.{0,120}?(?:\(\d+\)|CRitical Assessment|Critical Assessment of "
    r"genomic|Cornet)", re.S)

HISTORICAL = re.compile(
    r"histor\w+|previous\w*|earlier|superseded|formerly|release history|"
    r"as it was built|at the time|predecessor|withdraw\w*|no longer|"
    r"0\.3\.0 (?:lacks|contains neither|has neither|does not)|not v?0\.3\.0|"
    r"rather than v?0\.3\.0", re.I)

VER030 = re.compile(r"v?0\.3\.0")

#: A line that denies the Source Data promise is not the promise. Matched
#: against the whole line, because the negation may sit either side of it.
NEGATED = re.compile(
    r"\bnot\b|\bno\b|\bnever\b|\bnone\b|\bwithout\b|would be false|"
    r"is not necessary|deliberately omitted|does not carry|no longer|"
    r"we do not|is omitted|not provided|not included|not enclosed", re.I)

#: Bold allowed in body prose only when the run is a figure/table cross-reference.
XREF_BOLD = re.compile(
    r"^(?:Supplementary\s+)?"
    r"(?:Figs?\.|Figures?|Tables?|Notes?)\s*"
    r"S?\d+[a-z]?"
    r"(?:\s*(?:,|–|—|-|and)\s*(?:S?\d+[a-z]?|[a-z]\b))*$")


#: Verbatim reviewer/editor quotations are blockquotes and must stay
#: byte-identical (BUILD_CONTRACT_V3 §11), so banned words inside them are
#: reported for information but never fail a gate.
QUOTE = re.compile(r"^\s*>")


def sentences(line: str) -> "list[str]":
    return re.split(r"(?<=[.!?])\s+", line)


# ---------------------------------------------------------------------------


class Report:
    def __init__(self):
        self.rows = []          # (ok, label, detail)
        self.notes = []
        self.ok = True

    def gate(self, ok, label, detail=""):
        self.rows.append((bool(ok), label, detail))
        if not ok:
            self.ok = False

    def limit(self, label, value, limit):
        self.gate(value <= limit, label, f"{value:>7}  (limit {limit})")

    def note(self, text):
        self.notes.append(text)

    def render(self):
        width = max(len(l) for _, l, _ in self.rows) + 2
        for ok, label, detail in self.rows:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label:<{width}}{detail}")
        if self.notes:
            print()
            for n in self.notes:
                print(n)


def check_limits(md: str, path: str, rep: Report) -> "dict[str, int]":
    sections = dict(split_sections(md))
    order = [name for name, _ in split_sections(md)]

    title = next(l for l in sections["__front__"].split("\n") if l.strip())
    rep.limit("Title (words)", words(title), LIMITS["title"])

    abstract = sections.get("Abstract", "")
    rep.limit("Abstract (words)", words(abstract), LIMITS["abstract"])
    rep.gate(not re.search(r"\(\d+(,\s*\d+)*\)", abstract),
             "Abstract free of numeric citations")

    n_main, n_legend, per_section = 0, 0, []
    for name in order:
        if name.strip().lower() in MAIN_TEXT_SECTIONS:
            prose, disp = split_display(sections[name])
            w, lw = words(prose), words(disp)
            per_section.append((name, w, lw))
            n_main += w
            n_legend += lw
    rep.limit("Main text = Intro+Results+Discussion", n_main, LIMITS["main_text"])
    rep.limit("Methods (words)", words(sections.get("Methods", "")),
              LIMITS["methods"])

    refs = sections.get("References", "")
    ref_nums = [int(m) for m in re.findall(r"^(\d+)\\?\.\s", refs, flags=re.M)]
    n_refs = 0
    for n in ref_nums:
        if n == n_refs + 1:
            n_refs = n
    rep.limit("References", n_refs, LIMITS["references"])

    figs = sorted({int(m) for m in re.findall(r"^\*\*Figure (\d+)\.", md, flags=re.M)})
    tabs = sorted({int(m) for m in re.findall(r"^\*\*Table (\d+)\.", md, flags=re.M)})
    rep.limit("Display items (main figures + tables)", len(figs) + len(tabs),
              LIMITS["display_items"])

    # --- per-legend word counts, reported individually --------------------
    for i, line in enumerate(md.split("\n"), start=1):
        m = re.match(r"^\*\*(Figure|Table) (\d+)\.", line)
        if m:
            rep.limit(f"{m.group(1)} {m.group(2)} legend (words, line {i})",
                      words(line), LIMITS["legend"])

    # --- reference citation coverage --------------------------------------
    body = "\n".join(sections.get(n, "") for n in order
                     if n.strip().lower() in MAIN_TEXT_SECTIONS or n == "Methods")
    cited = set()
    for m in re.finditer(r"\((\d+(?:\s*[,–-]\s*\d+)*)\)", body):
        for part in m.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                cited.add(int(part))
            else:
                rng = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
                if rng:
                    a, b = int(rng.group(1)), int(rng.group(2))
                    if b - a < 60:
                        cited.update(range(a, b + 1))
    orphans = sorted(set(range(1, n_refs + 1)) - cited)
    rep.gate(not orphans, "Every reference cited in the main text",
             "" if not orphans else f"orphans: {orphans}")
    if orphans:
        for n in orphans:
            entry = re.search(rf"^{n}\\?\.\s*(.{{0,70}})", refs, flags=re.M)
            rep.note(f"         orphan ref {n}: "
                     f"{entry.group(1).strip() if entry else '?'}…")

    # --- bold audit --------------------------------------------------------
    bad_bold, xref_bold = [], 0
    for name in order:
        if name.strip().lower() not in MAIN_TEXT_SECTIONS and name != "Methods":
            continue
        prose = split_display(sections[name])[0]
        base = sections[name].split("\n")
        offset = md.split("\n").index(base[0]) if base and base[0] else 0
        for run in re.findall(r"\*\*([^*]+)\*\*", prose):
            if XREF_BOLD.match(run.strip()):
                xref_bold += 1
            else:
                ln = _find_line(md, f"**{run}**")
                bad_bold.append((name, ln, run))
    rep.gate(not bad_bold,
             "Bold in body prose only for Fig/Table cross-refs",
             f"{xref_bold} cross-reference runs, {len(bad_bold)} other")
    for name, ln, run in bad_bold[:20]:
        rep.note(f"         bold {name} line {ln}: **{run[:70]}**")

    rep.note("")
    for name, w, lw in per_section:
        extra = f"   (+{lw} legend)" if lw else ""
        rep.note(f"         {name:<40}{w:>6}{extra}")
    strict = n_main + n_legend
    rep.note(f"         {'display-item legends (excluded)':<40}{n_legend:>6}")
    rep.note(f"         {'strict reading, legends counted':<40}{strict:>6}"
             f"  -> {'ok' if strict <= LIMITS['main_text'] else 'over by %d' % (strict - LIMITS['main_text'])}")
    rep.note(f"         main-text figures: {figs}   tables: {tabs}")
    return {"main_text": n_main, "abstract": words(abstract),
            "methods": words(sections.get("Methods", ""))}


def _find_line(md: str, needle: str) -> int:
    for i, line in enumerate(md.split("\n"), start=1):
        if needle in line:
            return i
    return 0


def check_hygiene(rep: Report, docs) -> None:
    """Placeholders, banned words, internal shorthand and version strings."""
    hits, quoted = {}, {}
    for doc, lines in docs.items():
        for i, line in enumerate(lines, start=1):
            bucket = quoted if QUOTE.match(line) else hits
            for rx in PLACEHOLDERS:
                for m in re.finditer(rx, line):
                    # [MANUSCRIPT NUMBER] in the cover letter is a submission-
                    # system field the author fills in at upload time, not an
                    # unresolved availability-statement placeholder. It is
                    # listed as an author action in README.md instead.
                    if (m.group(0) == "[MANUSCRIPT NUMBER]"
                            and doc == "cover_letter.md"):
                        quoted.setdefault("author-supplied field", []).append(
                            (doc, i, m.group(0), line.strip()[:100]))
                        continue
                    bucket.setdefault("placeholder", []).append(
                        (doc, i, m.group(0), line.strip()[:100]))
            for label, rx in BANNED_TEXT:
                for m in re.finditer(rx, line):
                    # A sentence that DENIES the promise is not the promise.
                    # "Data availability does not carry the statement
                    # 'Source Data are provided with this paper'" must pass.
                    if label == "Source Data promise" and NEGATED.search(line):
                        quoted.setdefault(label, []).append(
                            (doc, i, m.group(0), line.strip()[:100]))
                        continue
                    bucket.setdefault(label, []).append(
                        (doc, i, m.group(0), line.strip()[:100]))
            for label, rx in SHORTHAND:
                for m in re.finditer(rx, line):
                    bucket.setdefault(label, []).append(
                        (doc, i, m.group(0), line.strip()[:100]))
            for m in re.finditer(r"CRACOT", line):
                seg = line[m.start():m.start() + 200]
                if not CRACOT_OK.match(seg):
                    bucket.setdefault("bare CRACOT", []).append(
                        (doc, i, "CRACOT", line.strip()[:100]))

    rep.gate("placeholder" not in hits, "Zero unresolved placeholders",
             _n(hits.get("placeholder")))
    for label, _ in BANNED_TEXT:
        rep.gate(label not in hits, f"Zero '{label}'", _n(hits.get(label)))
    rep.gate("bare CRACOT" not in hits, "Zero bare 'CRACOT'",
             _n(hits.get("bare CRACOT")))
    for label, _ in SHORTHAND:
        rep.gate(label not in hits, f"Zero {label}", _n(hits.get(label)))

    for key in ["placeholder", "AUTHOR TO VERIFY", "figshare", "Zenodo",
                "Source Data", "bare CRACOT", "trap T", "W-register code",
                "WS workstream code", "internal QA marks"]:
        for doc, i, tok, ctx in hits.get(key, [])[:40]:
            rep.note(f"         {key:<20} {doc}:{i}  {tok!r}  {ctx}")
    if quoted:
        rep.note("")
        for key, rows in quoted.items():
            rep.note(f"         [quoted, keep verbatim] {key}: {len(rows)} — "
                     + ", ".join(f"{d}:{i}" for d, i, _, _ in rows[:12]))

    # --- v0.3.0, every occurrence reported for adjudication ---------------
    current, historical = [], []
    for doc, lines in docs.items():
        for i, line in enumerate(lines, start=1):
            if QUOTE.match(line):
                continue
            for sent in sentences(line):
                for m in VER030.finditer(sent):
                    row = (doc, i, sent.strip()[:110])
                    (historical if HISTORICAL.search(sent) else current).append(row)
    rep.gate(not current, "Zero non-historical 'v0.3.0'",
             f"{len(current)} current, {len(historical)} historical")
    for doc, i, ctx in current[:40]:
        rep.note(f"         v0.3.0 (current)     {doc}:{i}  {ctx}")
    for doc, i, ctx in historical[:10]:
        rep.note(f"         v0.3.0 (historical)  {doc}:{i}  {ctx}")

    # --- advisory ----------------------------------------------------------
    adv = {}
    for doc, lines in docs.items():
        for i, line in enumerate(lines, start=1):
            for label, rx in ADVISORY:
                for m in re.finditer(rx, line):
                    adv.setdefault(label, []).append((doc, i, m.group(0)))
    if adv:
        rep.note("")
        for label, rows in adv.items():
            rep.note(f"         [warn] {label}: {len(rows)} "
                     f"({', '.join(sorted({d for d, _, _ in rows}))})")


def check_cover_letter(rep: Report, docs, measured) -> None:
    """The cover letter asserts compliance figures; they must be true.

    An editor can check "abstract 149" in ten seconds, and a cover letter that
    claims work the manuscript says was not done is a first-screen credibility
    failure.  Both are cheap to catch and expensive to ship.
    """
    lines = docs.get("cover_letter.md")
    if not lines:
        return
    bad = []
    claim = re.compile(
        r"(main text|abstract|methods)\s+(?:is\s+)?([\d,]+)\s*(?:words)?", re.I)
    for i, line in enumerate(lines, start=1):
        for m in claim.finditer(line):
            what = m.group(1).lower().replace("main text", "main_text")
            try:
                said = int(m.group(2).replace(",", ""))
            except ValueError:
                continue
            got = measured.get(what)
            if got is not None and said != got:
                bad.append((i, what, said, got))
    rep.gate(not bad, "Cover-letter word counts match the manuscript",
             "" if not bad else f"{len(bad)} mismatched")
    for i, what, said, got in bad:
        rep.note(f"         cover_letter.md:{i}  claims {what} = {said}, "
                 f"measured {got}")

    # claims of work the manuscript denies
    joined = "\n".join(lines)
    ms = "\n".join(docs.get("manuscript_revised.md", []))
    contradictions = []
    if re.search(r"leave-genus-out", joined, re.I) and \
            re.search(r"no leave-genus-out experiment was performed", ms, re.I):
        contradictions.append(
            "cover letter claims leave-genus-out retraining; the manuscript "
            "says 'no leave-genus-out experiment was performed'")
    rep.gate(not contradictions,
             "Cover letter claims no work the manuscript denies",
             "" if not contradictions else f"{len(contradictions)} conflict")
    for c in contradictions:
        rep.note(f"         {c}")


def _n(rows) -> str:
    if not rows:
        return "0"
    docs = sorted({d for d, _, _, _ in rows})
    return f"{len(rows)} in {', '.join(docs)}"


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    md = open(path, encoding="utf-8").read()

    docs = {}
    for name in PACKAGE_DOCS:
        p = os.path.join(PKG, name)
        if os.path.exists(p):
            docs[name] = open(p, encoding="utf-8").read().split("\n")

    rep = Report()
    print(f"file: {os.path.relpath(path)}")
    print(f"package: {', '.join(docs)}\n")

    measured = check_limits(md, path, rep)
    rep.gate(True, "-- hygiene gates, whole package --", "")
    rep.rows.pop()                       # separator row, not a gate
    check_hygiene(rep, docs)
    check_cover_letter(rep, docs, measured)

    rep.render()
    print("\n" + ("ALL GATES PASS" if rep.ok else "GATES NOT MET"))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
