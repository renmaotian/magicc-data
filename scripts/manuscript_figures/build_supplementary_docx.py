#!/usr/bin/env python
"""Build ``supplementary_revised.docx`` by rendering ``supplementary_revised.md``.

The Supplementary has diverged too far from the submitted v4 file for the old
change-map approach (edit ``1st_submission/supplementary_v4.docx`` in place with
``supplementary_change_map.md``) to be safe: it now carries 18 tables, 17
figures, a Supplementary Methods section and six Supplementary Notes that have
no v4 counterpart.  This script therefore renders the markdown *directly* and
takes only the document furniture from v4:

  * ``styles.xml`` (Title / Heading 1-3 / Normal, Arial 11 pt),
  * the A4 ``sectPr`` with 1 in margins, header and footer parts,
  * ``theme``, ``fontTable``, ``numbering`` and ``settings``,
  * the ``w:tblPr`` of a v4 table -- borders, cell margins, ``tblLook`` -- which
    is cloned for every one of the 88 markdown tables, and
  * the five v4 inline-image paragraphs, whose PNG blob is swapped for Figures
    S1-S5; Figures S6-S17 are added as new pictures.

So the look is unchanged and nothing in the body is inherited by accident.

Layout rules enforced here (author instruction):

  * every figure caption sits immediately BELOW its image;
  * every table legend sits immediately BELOW its table;
  * no separate "Figure legends" / "Table legends" section.

Verification (all assertions, the script exits non-zero on any failure):
reopen the file; count figures and tables; diff every table cell and every
number against the markdown; confirm each of the 17 captions directly follows
its own image and each table legend directly follows its own table; confirm no
raw markdown survives; confirm nothing is painted past the right margin; render
to PDF and report the page count and page size.

Run:  /path/to/conda/bin/python build_supplementary_docx.py
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

import docx
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Emu, Inches, Pt
from docx.table import Table
from docx.text.paragraph import Paragraph

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NC = os.path.join(ROOT, "nature_communications")
V4_DOCX = os.path.join(NC, "1st_submission", "supplementary_v4.docx")
HERE = os.path.join(NC, "resubmission2")
MD = os.path.join(HERE, "supplementary_revised.md")
FIGDIR = os.path.join(HERE, "supp_figures")
OUT = os.path.join(HERE, "supplementary_revised.docx")

N_FIGURES = 17

# --------------------------------------------------------------------------
# page geometry (A4, 1 in margins) -- read back from the file, not assumed
# --------------------------------------------------------------------------
USABLE_TWIPS = None          # filled in from sectPr
TABLE_TOTAL_TWIPS = 9000     # matches the v4 tables (8,916-9,012 twips)
IMG_MAX_W_IN = 6.25
IMG_MAX_H_IN = 8.60

# ==========================================================================
# markdown parsing
# ==========================================================================


def split_row(line: str) -> list:
    """Split a markdown table row on unescaped pipes."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    out, buf, i = [], [], 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            buf.append("|")
            i += 2
            continue
        if ch == "|":
            out.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    out.append("".join(buf).strip())
    return out


SEP_RE = re.compile(r"^\|[\s:\-|]+\|?\s*$")


def parse_markdown(path: str) -> list:
    """Return a flat list of blocks: (kind, payload)."""
    lines = open(path, encoding="utf-8").read().split("\n")
    blocks, i, n = [], 0, len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()

        if not s:
            i += 1
            continue

        if re.fullmatch(r"-{3,}", s):
            blocks.append(("hr", None))
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            blocks.append(("h", (len(m.group(1)), m.group(2).strip())))
            i += 1
            continue

        if s.startswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                raw = lines[i].strip()
                if not SEP_RE.match(raw):
                    rows.append(split_row(raw))
                i += 1
            if rows:
                ncols = max(len(r) for r in rows)
                rows = [r + [""] * (ncols - len(r)) for r in rows]
                blocks.append(("table", rows))
            continue

        if s.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]).strip())
                i += 1
            blocks.append(("quote", " ".join(x for x in buf if x)))
            continue

        if re.match(r"^\s*-\s+", line):
            buf = [re.sub(r"^\s*-\s+", "", line).strip()]
            i += 1
            while i < n:
                nxt = lines[i]
                if not nxt.strip():
                    break
                if re.match(r"^\s*-\s+", nxt) or nxt.strip().startswith(("|", ">", "#")):
                    break
                buf.append(nxt.strip())
                i += 1
            blocks.append(("bullet", " ".join(buf)))
            continue

        buf = [s]
        i += 1
        while i < n:
            nxt = lines[i]
            if not nxt.strip():
                break
            if nxt.strip().startswith(("|", ">", "#")) or re.match(r"^\s*-\s+", nxt) \
                    or re.fullmatch(r"-{3,}", nxt.strip()):
                break
            buf.append(nxt.strip())
            i += 1
        blocks.append(("p", " ".join(buf)))
    return blocks


# ==========================================================================
# inline markdown -> runs
# ==========================================================================

EMDASH_RE = re.compile(r"--")


def parse_inline(text: str) -> list:
    """Return [(text, bold, italic, code), ...].  Well-formed markdown assumed;
    an unbalanced marker degrades to a literal character rather than raising."""
    runs, buf = [], []
    bold = italic = False
    i, n = 0, len(text)

    def flush(code=False):
        if buf:
            t = "".join(buf)
            if not code:
                t = EMDASH_RE.sub("\u2013", t)
            runs.append((t, bold, italic, code))
            buf.clear()

    while i < n:
        ch = text[i]
        if ch == "`":
            j = text.find("`", i + 1)
            if j == -1:
                buf.append(ch)
                i += 1
                continue
            flush()
            code_txt = text[i + 1:j]
            if code_txt:
                runs.append((code_txt, bold, italic, True))
            i = j + 1
            continue
        if text.startswith("**", i):
            flush()
            bold = not bold
            i += 2
            continue
        if ch == "*":
            flush()
            italic = not italic
            i += 1
            continue
        buf.append(ch)
        i += 1
    flush()
    return [r for r in runs if r[0]]


def plain_text(text: str) -> str:
    return "".join(r[0] for r in parse_inline(text))


# ==========================================================================
# docx helpers
# ==========================================================================

CODE_FONT = "Consolas"
ZWSP = "​"
BREAK_AFTER = set("/_-,.:;{}()=|\\&+")
LONG_TOKEN = 14


# --- tokens soft_break() must leave alone ---------------------------------
# A zero-width space is invisible on the page but it is a real character in
# the text stream: it survives copy-paste, and journal production converts
# Word to XML, where stray format characters are a known source of
# corruption.  Two classes of token are therefore exempt.
_OPENERS = "([{<\u2018\u201c\"'"
_CLOSERS = ")]}>\u2019\u201d\"'.,;:!?"
_URL_PREFIXES = ("http://", "https://", "www.", "doi:")
_DOI_RE = re.compile(r"10\.\d{4,9}/")   # a real DOI prefix, not "10.59/54.42"
_HYPHEN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")


def _is_link(tok: str) -> bool:
    """A web address or DOI, however prose wrapped it in brackets or quotes.

    The Data availability and Code availability statements exist so that a
    reader can follow the link; a hidden character inside one stops it
    resolving once it is copied out of the PDF.  The longest link in either
    document is 51 characters wide -- 3.61 in set at 11 pt Arial, against a
    6.268 in text column -- so none of them needs help wrapping.
    """
    core = tok.lstrip(_OPENERS).lower()
    return core.startswith(_URL_PREFIXES) or _DOI_RE.match(core) is not None


def _is_hyphenated_word(tok: str) -> bool:
    """``cluster-bootstrap``, ``MAGICC-minus-CheckM2``: alphanumeric parts
    joined by hyphens (digits allowed, for ``CheckM2`` and ``v5``).

    Word and LibreOffice both break a line after a hyphen natively, so the
    helper would only add a second, hidden copy of a break opportunity the
    text already carries.
    """
    return _HYPHEN_WORD_RE.fullmatch(tok.strip(_OPENERS + _CLOSERS)) is not None


def soft_break(text: str) -> str:
    """Insert zero-width spaces inside over-long tokens.

    File paths and identifiers such as
    ``data/benchmarks/motivating_v2/set_B/{checkm2,...}_predictions.tsv`` are
    single unbreakable words; Word and LibreOffice both push them past the
    right margin rather than breaking them.  A zero-width space is an invisible
    break opportunity that both honour, and it does not change the visible
    text.

    Links and ordinary hyphenated compounds are exempt: see :func:`_is_link`
    and :func:`_is_hyphenated_word`.
    """
    out = []
    for tok in re.split(r"(\s+)", text):
        if (len(tok) > LONG_TOKEN and not tok.isspace()
                and not _is_link(tok) and not _is_hyphenated_word(tok)):
            buf, run_len = [], 0
            for ch in tok:
                buf.append(ch)
                run_len += 1
                if ch in BREAK_AFTER and run_len >= 6:
                    buf.append(ZWSP)
                    run_len = 0
            tok = "".join(buf)
        out.append(tok)
    return "".join(out)


def unbreakable_len(text: str) -> int:
    """Longest run of characters with no break opportunity."""
    parts = re.split(r"[\s​]+", soft_break(text))
    return max((len(p) for p in parts if p), default=1)


def set_run_font(run, name):
    rpr = run._r.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs"):
        rf.set(qn(a), name)
    rf.attrib.pop(qn("w:hint"), None)


def fill_runs(par, text, size_pt=None, force_bold=False, force_italic=False):
    """Replace the paragraph's runs with formatted runs parsed from markdown."""
    for r in list(par._p.findall(qn("w:r"))):
        par._p.remove(r)
    for hl in list(par._p.findall(qn("w:hyperlink"))):
        par._p.remove(hl)
    for txt, bold, italic, code in parse_inline(text):
        run = par.add_run(soft_break(txt))
        run.bold = True if (bold or force_bold) else None
        run.italic = True if (italic or force_italic) else None
        if code:
            set_run_font(run, CODE_FONT)
        if size_pt is not None:
            run.font.size = Pt(size_pt)
    return par


def new_paragraph(doc, style=None):
    p = OxmlElement("w:p")
    par = Paragraph(p, doc._body)
    if style is not None:
        par.style = doc.styles[style]
    return par


def set_indent(par, left_twips=0, hanging_twips=0, right_twips=0):
    ppr = par._p.get_or_add_pPr()
    ind = ppr.find(qn("w:ind"))
    if ind is None:
        ind = OxmlElement("w:ind")
        ppr.append(ind)
    ind.set(qn("w:left"), str(left_twips))
    if right_twips:
        ind.set(qn("w:right"), str(right_twips))
    if hanging_twips:
        ind.set(qn("w:hanging"), str(hanging_twips))


def set_spacing(par, before=None, after=None):
    ppr = par._p.get_or_add_pPr()
    sp = ppr.find(qn("w:spacing"))
    if sp is None:
        sp = OxmlElement("w:spacing")
        ppr.insert(0, sp)
    if before is not None:
        sp.set(qn("w:before"), str(before))
    if after is not None:
        sp.set(qn("w:after"), str(after))


def add_left_bar(par):
    ppr = par._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "12")
    left.set(qn("w:space"), "6")
    left.set(qn("w:color"), "808080")
    bdr.append(left)
    ppr.append(bdr)


# ---------------------------------------------------------------- tables ---


# --------------------------------------------------------------------------
# Text metrics.  Arial's advance widths are the Helvetica ones; using the real
# table instead of an average keeps caps-heavy headers such as "CheckM2" from
# wrapping in a column that the estimate said was wide enough.
# --------------------------------------------------------------------------
_HELV = {
    " ": 278, "!": 278, '"': 355, "#": 556, "$": 556, "%": 889, "&": 667,
    "'": 191, "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333,
    ".": 278, "/": 278, ":": 278, ";": 278, "<": 584, "=": 584, ">": 584,
    "?": 556, "@": 1015, "[": 278, "\\": 278, "]": 278, "^": 469, "_": 556,
    "`": 333, "{": 334, "|": 260, "}": 334, "~": 584,
    "A": 667, "B": 667, "C": 722, "D": 722, "E": 667, "F": 611, "G": 778,
    "H": 722, "I": 278, "J": 500, "K": 667, "L": 556, "M": 833, "N": 722,
    "O": 778, "P": 667, "Q": 778, "R": 722, "S": 667, "T": 611, "U": 722,
    "V": 667, "W": 944, "X": 667, "Y": 667, "Z": 611,
    "a": 556, "b": 556, "c": 500, "d": 556, "e": 556, "f": 278, "g": 556,
    "h": 556, "i": 222, "j": 222, "k": 500, "l": 222, "m": 833, "n": 556,
    "o": 556, "p": 556, "q": 556, "r": 333, "s": 500, "t": 278, "u": 556,
    "v": 500, "w": 722, "x": 500, "y": 500, "z": 500,
}
for _d in "0123456789":
    _HELV[_d] = 556
_DEFAULT_ADV = 600      # non-ASCII: en/em dash, +-, >=, superscripts, Greek
_BOLD_FACTOR = 1.07
_MONO_ADV = 550         # Consolas is 0.55 em for every glyph


def _adv(ch, bold, code):
    if code:
        w = _MONO_ADV
    else:
        w = _HELV.get(ch, _DEFAULT_ADV)
    return w * (_BOLD_FACTOR if bold else 1.0) / 1000.0


def md_metrics(cell_md, header=False):
    """(total width, longest unbreakable width) of a markdown fragment, in em."""
    stream = []          # (char, advance_em)
    for txt, bold, italic, code in parse_inline(cell_md):
        for ch in soft_break(txt):
            stream.append((ch, _adv(ch, bold or header, code)))
    total = sum(a for _, a in stream)
    longest = run = 0.0
    for ch, a in stream:
        if ch.isspace() or ch == ZWSP:
            run = 0.0
        else:
            run += a
            longest = max(longest, run)
    return total, longest


def text_em(text, bold=False):
    return sum(_adv(ch, bold, False) for ch in text)


def char_width(cell_text: str) -> int:
    return len(plain_text(cell_text))


CELL_PAD = 90            # twips: cell margins plus the two cell borders
MIN_COL_TWIPS = 480
TARGET_LINE_EM = 17.0    # a comfortable measure for a table cell, in em


def column_metrics(rows, size_pt):
    """(need, want) per column, in twips, at ``size_pt``."""
    ncols = len(rows[0])
    twips_per_em = size_pt * 20.0
    need, want = [], []
    for c in range(ncols):
        totals, longests = [], []
        for ri, r in enumerate(rows):
            t, l = md_metrics(r[c], header=(ri == 0))
            totals.append(t)
            longests.append(l)
        need.append(max(max(longests) * twips_per_em + CELL_PAD, MIN_COL_TWIPS))
        want.append(min(max(totals), TARGET_LINE_EM) * twips_per_em + CELL_PAD)
    return need, want


def allocate(total, need, want):
    """Split ``total`` proportionally to ``want`` but never below ``need``."""
    n = len(need)
    if sum(need) >= total:
        scale = total / sum(need)
        w = [x * scale for x in need]
    else:
        w = [None] * n
        free = set(range(n))
        remaining = float(total)
        while free:
            tot_w = sum(want[i] for i in free) or 1.0
            pinned = [i for i in free if remaining * want[i] / tot_w < need[i]]
            if not pinned:
                for i in free:
                    w[i] = remaining * want[i] / tot_w
                break
            for i in pinned:
                w[i] = need[i]
                remaining -= need[i]
                free.discard(i)
    out = [int(round(x)) for x in w]
    k = max(range(n), key=lambda i: out[i])
    out[k] += total - sum(out)
    return out


def table_layout(rows, total=TABLE_TOTAL_TWIPS):
    """Choose the body font size and the column widths together.

    Wide tables start at 8 pt (requirement: they shrink rather than overflow);
    the size drops further only if the unbreakable minimum widths still do not
    fit.
    """
    ncols = len(rows[0])
    max_row_em = max(sum(md_metrics(c, header=(ri == 0))[0] for c in r)
                     for ri, r in enumerate(rows))
    if ncols >= 6:
        start = 8.0
    elif ncols == 5:
        start = 9.0
    else:
        start = 10.0
    if max_row_em > 145 and start > 8.0:
        start -= 1.0
    if max_row_em > 235:
        start = 8.0

    for size_pt in [x for x in (10.0, 9.0, 8.0, 7.0, 6.5) if x <= start]:
        need, want = column_metrics(rows, size_pt)
        if sum(need) <= total:
            return size_pt, allocate(total, need, want)
    size_pt = 6.5
    need, want = column_metrics(rows, size_pt)
    scale = total / sum(need)
    need = [x * scale for x in need]
    return size_pt, allocate(total, need, want)


def build_table(doc, tbl_el, rows):
    """Rebuild ``tbl_el`` (an existing or cloned w:tbl) from ``rows`` in place.

    Row and cell formatting are taken from the element's own first (header) and
    second (body) rows, so an original v4 table keeps exactly its own look and a
    cloned one inherits the v4 look.
    """
    trs = tbl_el.findall(qn("w:tr"))
    hdr_tpl = copy.deepcopy(trs[0])
    body_tpl = copy.deepcopy(trs[1]) if len(trs) > 1 else copy.deepcopy(trs[0])
    for tr in trs:
        tbl_el.remove(tr)

    def strip_to_one_cell(tr_tpl):
        tcs = tr_tpl.findall(qn("w:tc"))
        keep = copy.deepcopy(tcs[0])
        for tc in tr_tpl.findall(qn("w:tc")):
            tr_tpl.remove(tc)
        return tr_tpl, keep

    hdr_row_tpl, hdr_cell_tpl = strip_to_one_cell(hdr_tpl)
    body_row_tpl, body_cell_tpl = strip_to_one_cell(body_tpl)

    ncols = len(rows[0])
    size_pt, widths = table_layout(rows)

    # alignment per column: centre short columns, left-align prose columns
    aligns = []
    for c in range(ncols):
        longest = max(char_width(r[c]) for r in rows)
        aligns.append("center" if longest <= 20 else "left")

    # table properties: fixed layout at a width that cannot overflow the page
    tblpr = tbl_el.find(qn("w:tblPr"))
    tblw = tblpr.find(qn("w:tblW"))
    if tblw is None:
        tblw = OxmlElement("w:tblW")
        tblpr.insert(0, tblw)
    tblw.set(qn("w:w"), str(TABLE_TOTAL_TWIPS))
    tblw.set(qn("w:type"), "dxa")
    for old in tblpr.findall(qn("w:tblLayout")):
        tblpr.remove(old)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    look = tblpr.find(qn("w:tblLook"))
    if look is not None:
        look.addprevious(layout)
    else:
        tblpr.append(layout)

    grid = tbl_el.find(qn("w:tblGrid"))
    if grid is None:
        grid = OxmlElement("w:tblGrid")
        tblpr.addnext(grid)
    for gc in list(grid.findall(qn("w:gridCol"))):
        grid.remove(gc)
    for w in widths:
        gc = OxmlElement("w:gridCol")
        gc.set(qn("w:w"), str(w))
        grid.append(gc)

    for ri, row in enumerate(rows):
        is_hdr = ri == 0
        tr = copy.deepcopy(hdr_row_tpl if is_hdr else body_row_tpl)
        cell_tpl = hdr_cell_tpl if is_hdr else body_cell_tpl
        for ci, cell_md in enumerate(row):
            tc = copy.deepcopy(cell_tpl)
            tcpr = tc.find(qn("w:tcPr"))
            tcw = tcpr.find(qn("w:tcW"))
            if tcw is None:
                tcw = OxmlElement("w:tcW")
                tcpr.insert(0, tcw)
            tcw.set(qn("w:w"), str(widths[ci]))
            tcw.set(qn("w:type"), "dxa")
            # keep exactly one paragraph in the cell
            ps = tc.findall(qn("w:p"))
            for extra in ps[1:]:
                tc.remove(extra)
            par = Paragraph(ps[0], doc._body)
            fill_runs(par, cell_md, size_pt=size_pt, force_bold=is_hdr)
            ppr = par._p.get_or_add_pPr()
            for j in list(ppr.findall(qn("w:jc"))):
                ppr.remove(j)
            jc = OxmlElement("w:jc")
            jc.set(qn("w:val"), aligns[ci] if not is_hdr else "center")
            ppr.append(jc)
            tr.append(tc)
        tbl_el.append(tr)
    return tbl_el


# ---------------------------------------------------------------- images ---


def png_size(path):
    with open(path, "rb") as fh:
        head = fh.read(32)
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def fit(path):
    w, h = png_size(path)
    width_in = IMG_MAX_W_IN
    height_in = width_in * h / w
    if height_in > IMG_MAX_H_IN:
        height_in = IMG_MAX_H_IN
        width_in = height_in * w / h
    return Inches(width_in), Inches(height_in)


def replace_image_blob(doc, par_el, png_path):
    """Swap the PNG behind the inline shape in ``par_el`` and rescale it."""
    blips = par_el.findall(".//" + qn("a:blip"))
    assert len(blips) == 1, "expected exactly one image in the paragraph"
    rid = blips[0].get(qn("r:embed"))
    part = doc.part.related_parts[rid]
    with open(png_path, "rb") as fh:
        part._blob = fh.read()
    for attr in ("_image", "_sha1", "_filename"):
        if hasattr(part, attr):
            try:
                delattr(part, attr)
            except AttributeError:
                pass
    cx, cy = fit(png_path)
    for ext in par_el.findall(".//" + qn("wp:extent")):
        ext.set("cx", str(int(cx)))
        ext.set("cy", str(int(cy)))
    for ext in par_el.findall(".//" + qn("a:ext")):
        ext.set("cx", str(int(cx)))
        ext.set("cy", str(int(cy)))
    for docpr in par_el.findall(".//" + qn("wp:docPr")):
        name = os.path.basename(png_path)
        docpr.set("name", name)
        docpr.set("descr", name)
        if "title" in docpr.attrib:
            docpr.set("title", name)
    return rid


# ==========================================================================
# build
# ==========================================================================

TABLE_LABEL_RE = re.compile(r"^Table\s+(S\d+[a-z]?)\b")
FIGURE_LEAD_RE = re.compile(r"^\*\*Figure\s+S(\d+)\.\*\*")
TABLE_LEGEND_RE = re.compile(r"^\*\*Table\s+S(\d+[a-z]?)\.\*\*")


def build():
    global USABLE_TWIPS

    shutil.copyfile(V4_DOCX, OUT)
    doc = docx.Document(OUT)

    sect = doc.sections[0]
    USABLE_TWIPS = int(
        (sect.page_width - sect.left_margin - sect.right_margin) / 635
    )  # EMU -> twips
    assert TABLE_TOTAL_TWIPS <= USABLE_TWIPS, (
        f"table width {TABLE_TOTAL_TWIPS} exceeds usable {USABLE_TWIPS} twips"
    )

    body = doc.element.body
    sectPr = body.find(qn("w:sectPr"))

    # --- inventory the v4 body, then keep only the furniture --------------
    children = [c for c in body.iterchildren() if c is not sectPr]
    v4_paras = [c for c in children if c.tag == qn("w:p")]
    v4_tables = [c for c in children if c.tag == qn("w:tbl")]
    v4_images = [p for p in v4_paras if p.findall(".//" + qn("a:blip"))]
    before = dict(paragraphs=len(v4_paras), tables=len(v4_tables),
                  images=len(v4_images))
    assert len(v4_tables) == 15, len(v4_tables)
    assert len(v4_images) == 5, len(v4_images)

    title_p = v4_paras[0]                       # "Supplementary Information"
    subtitle_p = v4_paras[1]                    # the article title
    table_template = copy.deepcopy(v4_tables[0])

    for c in children:
        body.remove(c)

    def append(el):
        sectPr.addprevious(el)

    append(title_p)
    append(subtitle_p)

    # --- markdown ---------------------------------------------------------
    blocks = parse_markdown(MD)

    # the md repeats the article title (carried by the v4 Title paragraphs);
    # keep the author line
    if blocks and blocks[0][0] == "p" and "MAGICC:" in blocks[0][1]:
        authors = blocks[1][1] if len(blocks) > 1 and blocks[1][0] == "p" else None
        blocks = blocks[2:] if authors else blocks[1:]
    else:
        authors = None
    if authors:
        par = new_paragraph(doc, "Normal")
        fill_runs(par, authors)
        ppr = par._p.get_or_add_pPr()
        jc = OxmlElement("w:jc")
        jc.set(qn("w:val"), "center")
        ppr.append(jc)
        set_spacing(par, after=200)
        append(par._p)

    in_figures = False
    fig_seen = []
    md_table_count = 0

    def blank():
        append(new_paragraph(doc, "Normal")._p)

    for kind, payload in blocks:
        if kind == "hr":
            continue

        if kind == "h":
            level, text = payload
            style = {1: "Heading 1", 2: "Heading 2", 3: "Heading 3"}.get(
                level, "Heading 3")
            par = new_paragraph(doc, style)
            fill_runs(par, text)
            append(par._p)
            if plain_text(text).strip().lower() == "supplementary figures":
                in_figures = True
            elif level == 1:
                in_figures = False
            continue

        if kind == "p":
            text = payload
            fm = FIGURE_LEAD_RE.match(text) if in_figures else None
            if fm:
                # ---- image first, caption immediately below it -----------
                fig_no = int(fm.group(1))
                assert fig_no == len(fig_seen) + 1, (fig_no, fig_seen)
                fig_seen.append(fig_no)
                png = os.path.join(FIGDIR, f"Figure_S{fig_no}.png")
                assert os.path.exists(png), png
                if fig_no <= len(v4_images):
                    img_par_el = v4_images[fig_no - 1]
                    replace_image_blob(doc, img_par_el, png)
                    append(img_par_el)
                else:
                    ip = new_paragraph(doc, "Normal")
                    set_spacing(ip, before=240, after=120)
                    jc = OxmlElement("w:jc")
                    jc.set(qn("w:val"), "center")
                    ip._p.get_or_add_pPr().append(jc)
                    append(ip._p)
                    cx, cy = fit(png)
                    ip.add_run().add_picture(png, width=cx, height=cy)
                par = new_paragraph(doc, "Normal")
                fill_runs(par, text)
                set_spacing(par, after=240)
                append(par._p)
                blank()
                continue
            par = new_paragraph(doc, "Normal")
            fill_runs(par, text)
            set_spacing(par, after=120)
            append(par._p)
            continue

        if kind == "bullet":
            par = new_paragraph(doc, "Normal")
            fill_runs(par, "• " + payload)
            set_indent(par, left_twips=360, hanging_twips=200)
            set_spacing(par, after=60)
            append(par._p)
            continue

        if kind == "quote":
            par = new_paragraph(doc, "Normal")
            fill_runs(par, payload)
            set_indent(par, left_twips=360, right_twips=200)
            set_spacing(par, before=120, after=120)
            add_left_bar(par)
            append(par._p)
            continue

        if kind == "table":
            md_table_count += 1
            tbl_el = copy.deepcopy(table_template)
            build_table(doc, tbl_el, payload)
            append(tbl_el)
            blank()
            continue

        raise AssertionError(f"unhandled block {kind}")

    assert fig_seen == list(range(1, N_FIGURES + 1)), fig_seen
    doc.save(OUT)

    d2 = docx.Document(OUT)
    after = dict(paragraphs=len(d2.paragraphs), tables=len(d2.tables),
                 images=len(d2.inline_shapes))
    return before, after, md_table_count


# ==========================================================================
# verification
# ==========================================================================

NUM_RE = re.compile(r"[-−+]?\d[\d,]*\.?\d*")


def norm_num(s):
    return s.replace(",", "").replace("−", "-").rstrip(".")


def verify(before, after, md_table_count):
    print("=" * 74)
    print("VERIFICATION")
    print("=" * 74)

    doc = docx.Document(OUT)
    print(f"opened OK: {OUT}")
    print(f"  file size: {os.path.getsize(OUT) / 1e6:.2f} MB")
    print(f"  v4 furniture: paragraphs={before['paragraphs']:>4}  "
          f"tables={before['tables']:>3}  inline shapes={before['images']:>3}")
    print(f"  rendered    : paragraphs={after['paragraphs']:>4}  "
          f"tables={after['tables']:>3}  inline shapes={after['images']:>3}")

    assert after["images"] == N_FIGURES, after["images"]
    assert after["tables"] == md_table_count, (after["tables"], md_table_count)
    print(f"  OK  {N_FIGURES} inline shapes (Figures S1-S{N_FIGURES})")
    print(f"  OK  {after['tables']} Word tables == {md_table_count} markdown tables")

    # ---- headings --------------------------------------------------------
    blocks = parse_markdown(MD)
    md_heads = [plain_text(t) for k, (l, t) in
                ((k, p) for k, p in blocks if k == "h")]
    doc_paras = [p.text.replace(ZWSP, "").strip() for p in doc.paragraphs]
    doc_text = "\n".join(doc_paras)

    missing = [h for h in md_heads if h not in doc_text]
    print(f"  headings in markdown: {len(md_heads)}   missing from docx: "
          f"{len(missing)}")
    for m in missing:
        print("    MISSING:", m)
    assert not missing

    n_tab_heads = len([h for h in md_heads if TABLE_LABEL_RE.match(h)])
    print(f"  'Table Sn' headings present: {n_tab_heads}")

    # ---- every markdown body paragraph reached the document --------------
    md_paras = [plain_text(p) for k, p in blocks
                if k in ("p", "quote", "bullet")]
    body_missing = []
    for t in md_paras:
        t2 = " ".join(t.split())
        if not t2:
            continue
        if t2 not in " ".join(doc_text.split()):
            body_missing.append(t2[:90])
    print(f"  markdown body paragraphs: {len(md_paras)}   missing from docx: "
          f"{len(body_missing)}")
    for t in body_missing[:5]:
        print("    MISSING:", t)
    assert not body_missing

    # ---- table-by-table numeric diff -------------------------------------
    md_tables = [p for k, p in blocks if k == "table"]
    doc_tables = doc.tables
    assert len(md_tables) == len(doc_tables)
    bad_cells = bad_numbers = 0
    for ti, (mt, dt) in enumerate(zip(md_tables, doc_tables)):
        if len(mt) != len(dt.rows) or len(mt[0]) != len(dt.columns):
            print(f"    SHAPE MISMATCH table {ti}")
            bad_cells += 1
            continue
        for ri, row in enumerate(mt):
            for ci, cell in enumerate(row):
                want = plain_text(cell)
                got = dt.cell(ri, ci).text.replace(ZWSP, "")
                if " ".join(want.split()) != " ".join(got.split()):
                    bad_cells += 1
                    if bad_cells <= 10:
                        print(f"    CELL DIFF t{ti} r{ri} c{ci}: "
                              f"md={want!r} docx={got!r}")
                wn = [norm_num(x) for x in NUM_RE.findall(want)]
                gn = [norm_num(x) for x in NUM_RE.findall(got)]
                if wn != gn:
                    bad_numbers += 1
                    if bad_numbers <= 10:
                        print(f"    NUMERIC DIFF t{ti} r{ri} c{ci}: "
                              f"md={wn} docx={gn}")
    print(f"  cell text differences : {bad_cells}")
    print(f"  numeric differences   : {bad_numbers}")
    assert bad_cells == 0 and bad_numbers == 0

    # ---- raw markdown left over -----------------------------------------
    leftovers = {}
    scan = list(doc_paras)
    for t in doc.tables:
        for r in t.rows:
            for c in r.cells:
                scan.append(c.text.replace(ZWSP, ""))
    for pat, name in ((r"\*\*", "**"),
                      (r"^\s*\|.*\|\s*$", "| table row"),
                      (r"\|\s*-{3,}\s*\|", "|--- rule"),
                      (r"^#{1,6}\s", "#"), (r"\bTODO\b", "TODO"),
                      (r"\bTBD\b", "TBD"), (r"XXX", "XXX"),
                      (r"PLACEHOLDER", "PLACEHOLDER")):
        hits = [s for s in scan if re.search(pat, s, re.M)]
        if hits:
            leftovers[name] = hits[:3]
    print(f"  raw-markdown / TODO leftovers: "
          f"{ {k: len(v) for k, v in leftovers.items()} or 'none'}")
    assert not leftovers, leftovers

    # ---- author-fillable placeholders ------------------------------------
    for ph in ("[RELEASE]",):
        n_md = open(MD, encoding="utf-8").read().count(ph)
        n_doc = doc_text.count(ph)
        print(f"  placeholder {ph}: markdown {n_md}, docx {n_doc}")
        assert n_doc == n_md

    # ---- bold really is bold --------------------------------------------
    bold_runs = sum(1 for p in doc.paragraphs for r in p.runs if r.bold)
    hdr_bold = all(
        all(r.bold for p in c.paragraphs for r in p.runs) or not c.text.strip()
        for t in doc.tables for c in t.rows[0].cells)
    print(f"  bold runs in body paragraphs: {bold_runs};  every table header "
          f"row bold: {hdr_bold}")
    assert bold_runs > 0 and hdr_bold

    # ---- table width -----------------------------------------------------
    over = []
    for i, t in enumerate(doc.tables):
        g = t._tbl.find(qn("w:tblGrid"))
        w = sum(int(c.get(qn("w:w"))) for c in g.findall(qn("w:gridCol")))
        if w > USABLE_TWIPS:
            over.append((i, w))
    print(f"  usable text width: {USABLE_TWIPS} twips "
          f"({USABLE_TWIPS / 1440:.3f} in);  tables wider than that: {len(over)}")
    assert not over, over

    # ---- nothing wider than its container --------------------------------
    body_over = []
    for p in doc.paragraphs:
        if not p.text.strip():
            continue
        size = max((r.font.size.pt for r in p.runs if r.font.size), default=11.0)
        limit = USABLE_TWIPS
        ppr = p._p.find(qn("w:pPr"))
        if ppr is not None:
            i2 = ppr.find(qn("w:ind"))
            if i2 is not None:
                limit -= int(i2.get(qn("w:left")) or 0)
                limit -= int(i2.get(qn("w:right")) or 0)
        bold = any(r.bold for r in p.runs)
        worst = max((text_em(t, bold) for t in
                     re.split(r"[\s​]+", p.text) if t), default=0.0)
        if worst * size * 20.0 > limit:
            body_over.append((round(worst, 1), p.text[:70]))
    print(f"  body paragraphs with an unbreakable run wider than the text "
          f"column: {len(body_over)}")
    for w, t in body_over[:5]:
        print(f"    {w} em: {t!r}")
    assert not body_over

    cell_over = []
    for ti, t in enumerate(doc.tables):
        g = t._tbl.find(qn("w:tblGrid"))
        cw = [int(c.get(qn("w:w"))) for c in g.findall(qn("w:gridCol"))]
        for ri, row in enumerate(t.rows):
            for ci, cell in enumerate(row.cells):
                if ci >= len(cw):
                    continue
                size = max((r.font.size.pt for p in cell.paragraphs
                            for r in p.runs if r.font.size), default=10.0)
                bold = ri == 0 or any(r.bold for p in cell.paragraphs
                                      for r in p.runs)
                worst = max((text_em(x, bold) for x in
                             re.split(r"[\s​]+", cell.text) if x),
                            default=0.0)
                if worst * size * 20.0 > cw[ci] - CELL_PAD + 20:
                    cell_over.append((ti, ri, ci, round(worst, 1), cell.text[:40]))
    print(f"  table cells with an unbreakable run wider than their column: "
          f"{len(cell_over)}")
    for x in cell_over[:5]:
        print("   ", x)
    assert not cell_over

    sizes = sorted({round(r.font.size.pt, 1) for t in doc.tables
                    for row in t.rows for c in row.cells
                    for p in c.paragraphs for r in p.runs if r.font.size})
    print(f"  table body font sizes in use: {sizes} pt")

    # ---- images ----------------------------------------------------------
    max_w = max(s.width for s in doc.inline_shapes)
    max_h = max(s.height for s in doc.inline_shapes)
    usable_h = int((doc.sections[0].page_height
                    - doc.sections[0].top_margin
                    - doc.sections[0].bottom_margin))
    print(f"  widest image  {Emu(max_w).inches:.3f} in "
          f"(limit {USABLE_TWIPS / 1440:.3f} in)")
    print(f"  tallest image {Emu(max_h).inches:.3f} in "
          f"(page text height {Emu(usable_h).inches:.3f} in)")
    assert max_w <= USABLE_TWIPS * 635
    assert max_h <= usable_h
    for i, s in enumerate(doc.inline_shapes, start=1):
        w, h = png_size(os.path.join(FIGDIR, f"Figure_S{i}.png"))
        got, want = s.width / s.height, w / h
        assert abs(got - want) / want < 0.01, (i, got, want)
    print(f"  OK  aspect ratio preserved on all {N_FIGURES} images")

    # ---- every caption sits immediately below its own image --------------
    seq = []
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:tbl"):
            seq.append(("tbl", ""))
            continue
        if child.tag != qn("w:p"):
            continue
        p = Paragraph(child, doc)
        txt = p.text.replace(ZWSP, "").strip()
        if child.findall(".//" + qn("a:blip")):
            seq.append(("img", txt))
        elif re.match(r"^Figure S\d+\.", txt):
            seq.append(("cap", txt))
        elif re.match(r"^Table S\d+[a-z]?\.", txt):
            seq.append(("tleg", txt))
        elif txt:
            seq.append(("txt", txt))

    caps = [i for i, (k, _) in enumerate(seq) if k == "cap"]
    orphan_caps = [seq[i][1][:16] for i in caps
                   if i == 0 or seq[i - 1][0] != "img"]
    imgs = [i for i, (k, _) in enumerate(seq) if k == "img"]
    orphan_imgs = [i for i in imgs
                   if i + 1 >= len(seq) or seq[i + 1][0] != "cap"]
    order = [re.match(r"^Figure S(\d+)\.", seq[i][1]).group(1) for i in caps]
    print(f"  figure captions: {len(caps)}; images: {len(imgs)}; "
          f"captions not directly under an image: {len(orphan_caps)}; "
          f"images with no caption under them: {len(orphan_imgs)}")
    print(f"  caption order: S{', S'.join(order)}")
    assert len(caps) == len(imgs) == N_FIGURES
    assert not orphan_caps and not orphan_imgs
    assert order == [str(i) for i in range(1, N_FIGURES + 1)]

    tlegs = [i for i, (k, _) in enumerate(seq) if k == "tleg"]
    bad_tlegs = [seq[i][1][:20] for i in tlegs
                 if i == 0 or seq[i - 1][0] != "tbl"]
    print(f"  table legends ('Table Sn. ...'): {len(tlegs)}; not directly "
          f"below their table: {len(bad_tlegs)}")
    for b in bad_tlegs:
        print("    ORPHAN:", b)
    assert not bad_tlegs

    # ---- PDF -------------------------------------------------------------
    tmp = tempfile.mkdtemp(prefix="suppdocx_")
    r = subprocess.run(["/usr/bin/libreoffice", "--headless", "--convert-to",
                        "pdf", "--outdir", tmp, OUT],
                       capture_output=True, text=True, timeout=1800)
    pdf = os.path.join(tmp, os.path.basename(OUT).replace(".docx", ".pdf"))
    if not os.path.exists(pdf):
        print("  PDF CONVERSION FAILED:", r.stdout, r.stderr)
        raise SystemExit(1)
    info = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
    ps = re.search(r"^Page size:\s+(.*)$", info, re.M).group(1)
    print(f"  PDF rendered OK: {os.path.getsize(pdf) / 1e6:.2f} MB, "
          f"{pages} pages, page size {ps}")
    dims = re.match(r"([\d.]+) x ([\d.]+) pts", ps)
    assert dims and abs(float(dims.group(1)) - 595.3) < 0.1 \
        and abs(float(dims.group(2)) - 841.9) < 0.1, ps
    print(f"  page size confirmed A4: {dims.group(1)} x {dims.group(2)} pts")

    bbox = subprocess.run(["pdftotext", "-bbox", pdf, "-"],
                          capture_output=True, text=True).stdout
    right_limit = 595.304 - 72.0 + 1.5
    page_no, worst, offenders = 0, 0.0, []
    for line in bbox.split("\n"):
        if "<page " in line:
            page_no += 1
        m = re.search(r'<word xMin="([\d.]+)" yMin="([\d.]+)" '
                      r'xMax="([\d.]+)" yMax="([\d.]+)">(.*)</word>', line)
        if not m:
            continue
        xmax = float(m.group(3))
        if xmax > right_limit:
            worst = max(worst, xmax)
            offenders.append((page_no, round(xmax, 1), m.group(5)[:40]))
    print(f"  words painted past the right margin ({right_limit - 1.5:.0f} pt): "
          f"{len(offenders)}"
          + (f"   worst xMax {worst:.1f} pt" if offenders else ""))
    for o in offenders[:8]:
        print(f"    page {o[0]}  xMax {o[1]}  {o[2]!r}")
    assert not offenders, "content overflows the page width"
    shutil.copyfile(pdf, OUT.replace(".docx", ".pdf"))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  PDF kept at {OUT.replace('.docx', '.pdf')}")

    print("=" * 74)
    print("ALL CHECKS PASSED")
    return pages


if __name__ == "__main__":
    b, a, ntab = build()
    verify(b, a, ntab)
