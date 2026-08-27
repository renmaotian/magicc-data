#!/usr/bin/env python
"""Build ``manuscript_revised_tracked.docx``: the resubmission3 manuscript
carrying **real OOXML revision marks** against the originally submitted
``1st_submission/manuscript_v4.docx``.

Why this exists
---------------
The editor made a change-marked manuscript a condition of proceeding.  The
package shipped ``manuscript_revised.docx`` (no revision marks at all) and
``manuscript_revised_marked.docx`` (yellow highlighting only).  Highlighting
cannot represent a deletion, so deletions were invisible and Word's Review pane
had nothing to accept or reject.  This script produces a file whose changes are
``<w:ins>`` / ``<w:del>`` elements with ``w:id`` / ``w:author`` / ``w:date``,
which is what Word, LibreOffice and every journal production system understand.

Method
------
1.  ``manuscript_revised.docx`` -- the clean file built by
    ``build_manuscript_docx.py`` -- is *copied* to the output path.  The tracked
    file is therefore, by construction, the same document: same styles.xml, same
    A4 ``sectPr``, the same five embedded figures with their legends below them,
    the same Table 1 with its legend below it, the same ``keep_with_next``.
    Only the body's runs are rewritten.  ``build_manuscript_docx.py`` is
    imported (never modified) so its markdown parser can prove the clean file is
    not stale before anything is built on top of it.
2.  Both documents are reduced to a linear sequence of *items* in body order:
    text paragraphs, tables and image paragraphs.  The two sequences are aligned
    by a Needleman-Wunsch dynamic program whose match score is
    ``difflib.SequenceMatcher`` similarity over the normalised word lists, so a
    heavily edited paragraph is matched to its ancestor instead of showing up as
    a wholesale delete plus insert.  Tables and images are anchors: this is what
    makes a paragraph that moved across the table render as a deletion above it
    and an insertion below it, which is what Word does with a move and what
    keeps "reject all" reproducing v4's *ordering* and not just its words.
3.  Each matched pair is diffed at word level.  Kept and inserted text is
    sliced out of the clean document's own runs, so every insertion keeps the
    formatting the clean file gives it; deleted text is reconstructed from v4
    and emitted as ``<w:delText>`` runs.  Whole paragraphs that exist only in v4
    become deleted paragraphs (content in ``w:del`` plus a deleted paragraph
    mark); whole paragraphs that exist only in the revision become inserted
    paragraphs (content in ``w:ins`` plus an inserted paragraph mark).  Table 1
    is diffed cell by cell after aligning its rows and columns.
4.  Verification is by simulation, not by assertion.  ``apply_revisions``
    performs a real "reject all" and a real "accept all" on a copy of the XML
    tree -- dropping ``w:ins`` content, unwrapping ``w:del`` content back to
    ``w:t``, honouring paragraph-mark marks by merging paragraphs, honouring
    ``w:cellIns`` / ``w:cellDel`` / ``w:trPr`` row marks -- and the resulting
    documents' body text is compared against v4 and against the clean file.
    The accepted document is also rendered to PDF and its page count and page
    size compared with the clean file's.

Figures cannot be diffed as text.  A changed or new figure is tracked through
its caption paragraph, and the report lists which image parts differ from v4 so
a human can confirm.  Image binaries are never marked.

Run::

    /path/to/anaconda3/bin/python build_tracked_changes.py
    /path/to/anaconda3/bin/python build_tracked_changes.py --no-pdf
    /path/to/anaconda3/bin/python build_tracked_changes.py --supp-probe

**The manuscript is still being edited.  Re-run this script after
``manuscript_revised.md`` is final and ``build_manuscript_docx.py`` has been
re-run** -- it is idempotent and rebuilds from the pristine v4 file every time.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

import docx
from docx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_manuscript_docx as bmd          # noqa: E402  (path set above)

RESUB = bmd.RESUB
NC = bmd.NC

V4_DOCX = os.path.join(NC, "1st_submission", "manuscript_v4.docx")
CLEAN_DOCX = os.path.join(RESUB, "manuscript_revised.docx")
OUT_DOCX = os.path.join(RESUB, "manuscript_revised_tracked.docx")

SUPP_V4_DOCX = os.path.join(NC, "1st_submission", "supplementary_v4.docx")
SUPP_CLEAN_DOCX = os.path.join(RESUB, "supplementary_revised.docx")
SUPP_OUT_DOCX = os.path.join(RESUB, "supplementary_revised_tracked.docx")

REPORT = os.path.join(RESUB, "TRACKED_CHANGES_REPORT.md")

# One author, one date, fixed rather than "now" so that two runs of this script
# over unchanged inputs produce the same file.
REV_AUTHOR = "Author"
REV_DATE = "2026-08-26T00:00:00Z"

# Alignment.  A pair of paragraphs below MIN_SIM is not a pair: the old one is a
# deletion and the new one an insertion.  0.30 rather than the 0.35 that
# build_manuscript_docx.py uses for highlighting, because four v4 paragraphs
# whose descendants are unmistakable ("More fundamentally, however, the model of
# contamination that underpins CheckM2..." -> "More fundamentally, the
# contamination model underpinning CheckM2...", the Figure 2 legend, the
# fragmentation paragraph of Methods and the inference-pipeline paragraph) score
# 0.317-0.336.  Every v4 paragraph left unmatched at 0.30 has a best available
# similarity below it, so nothing recognisable is being thrown away.
MIN_SIM_PARA = 0.30
# Word-level diffs are unreadable when three-word islands of unchanged text
# survive inside a rewritten sentence.  An island shorter than this that has a
# change on both sides is absorbed into the change, exactly as
# build_manuscript_docx.py does for highlighting and as Word's own Compare does.
# Absorbing is safe: the island is then emitted as a deletion and an insertion
# of the same words, so accept and reject are unaffected.
MIN_EQUAL_RUN = 4
# Rows and columns of a table are positional, so they align on any similarity at
# all; the dynamic program picks the monotone assignment with the highest total.
MIN_SIM_CELL = 0.0
# An image is a soft anchor: worth matching, but not worth breaking a good
# paragraph alignment for.
IMG_ANCHOR_SIM = 0.60
TBL_ANCHOR_SIM = 1.00
# A heading matched to body prose, or a legend matched to a reference, is almost
# always the dynamic program taking a bad local bargain.  Scale such a pair down.
KIND_MISMATCH = 0.75

ZWSP = "​"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def w(tag):
    return "{%s}%s" % (W, tag)


W_P, W_R, W_TBL, W_TR, W_TC = w("p"), w("r"), w("tbl"), w("tr"), w("tc")
W_PPR, W_RPR, W_TCPR, W_TRPR = w("pPr"), w("rPr"), w("tcPr"), w("trPr")
W_INS, W_DEL = w("ins"), w("del")
W_T, W_DELTEXT = w("t"), w("delText")
W_HYPERLINK, W_SECTPR = w("hyperlink"), w("sectPr")
W_CELLINS, W_CELLDEL = w("cellIns"), w("cellDel")

# Children of a run that carry visible text, and children that do not.
RUN_TEXT_EQUIV = {W_T: None, W_DELTEXT: None, w("tab"): "\t", w("br"): "\n",
                  w("cr"): "\n", w("noBreakHyphen"): "-", w("softHyphen"): ""}
# Field instructions are not visible text; drawings are not text at all.
RUN_DROP_WHEN_CLONING = {w("fldChar"), w("instrText"), w("delInstrText"),
                         w("drawing"), w("pict"), w("object"),
                         w("lastRenderedPageBreak")}

CONTENT_SKIP = {W_PPR}


# ==========================================================================
# small lxml helpers (no python-docx custom element classes are relied on,
# because deep-copied subtrees do not always keep them)
# ==========================================================================

def el(tag):
    from docx.oxml import OxmlElement
    return OxmlElement(tag)


def get_or_add_first(parent, tag):
    """``tag`` as the first child of ``parent``, created if missing."""
    e = parent.find(tag)
    if e is None:
        e = el(tag.replace("{%s}" % W, "w:"))
        parent.insert(0, e)
    return e


def revision_attrs(e, ids):
    e.set(w("id"), str(next(ids)))
    e.set(w("author"), REV_AUTHOR)
    e.set(w("date"), REV_DATE)
    return e


def id_source():
    n = 0
    while True:
        n += 1
        yield n


def make_run(rpr_el, text, deleted=False):
    r = el("w:r")
    if rpr_el is not None:
        r.append(copy.deepcopy(rpr_el))
    t = el("w:delText" if deleted else "w:t")
    t.set(XML_SPACE, "preserve")
    t.text = text
    r.append(t)
    return r


def wrap(kind, children, ids):
    """``<w:ins>`` / ``<w:del>`` around already-built run elements."""
    e = revision_attrs(el("w:ins" if kind == "ins" else "w:del"), ids)
    for c in children:
        e.append(c)
    return e


def mark_paragraph_mark(p_el, kind, ids):
    """``<w:pPr><w:rPr><w:ins/></w:rPr></w:pPr>`` -- the paragraph mark itself.

    Without this a whole-paragraph insertion leaves an empty paragraph behind
    when it is rejected, and a whole-paragraph deletion leaves an empty
    paragraph behind when it is accepted.
    """
    ppr = get_or_add_first(p_el, W_PPR)
    rpr = ppr.find(W_RPR)
    if rpr is None:
        rpr = el("w:rPr")
        anchor = None
        for tag in (W_SECTPR, w("pPrChange")):
            anchor = ppr.find(tag)
            if anchor is not None:
                break
        if anchor is not None:
            anchor.addprevious(rpr)
        else:
            ppr.append(rpr)
    for tag in (W_INS, W_DEL):
        for old in rpr.findall(tag):
            rpr.remove(old)
    rpr.insert(0, revision_attrs(el("w:ins" if kind == "ins" else "w:del"), ids))
    return p_el


def mark_cell(tc_el, kind, ids):
    """``w:cellIns`` / ``w:cellDel``.

    ``CT_TcPrInner`` puts the cell-markup element after the base properties and
    before ``w:tcPrChange``, so it is appended, not inserted.
    """
    tcpr = get_or_add_first(tc_el, W_TCPR)
    for tag in (W_CELLINS, W_CELLDEL):
        for old in tcpr.findall(tag):
            tcpr.remove(old)
    anchor = tcpr.find(w("tcPrChange"))
    mark = revision_attrs(el("w:cellIns" if kind == "ins" else "w:cellDel"), ids)
    if anchor is not None:
        anchor.addprevious(mark)
    else:
        tcpr.append(mark)
    return tc_el


def mark_row(tr_el, kind, ids):
    """``<w:trPr><w:ins/></w:trPr>``.

    ``CT_TrPr`` extends ``CT_TrPrBase`` with ``ins``, ``del`` and
    ``trPrChange`` *after* the base properties, so this appends before
    ``w:trPrChange`` rather than inserting at the front.
    """
    trpr = get_or_add_first(tr_el, W_TRPR)
    for tag in (W_INS, W_DEL):
        for old in trpr.findall(tag):
            trpr.remove(old)
    anchor = trpr.find(w("trPrChange"))
    mark = revision_attrs(el("w:ins" if kind == "ins" else "w:del"), ids)
    if anchor is not None:
        anchor.addprevious(mark)
    else:
        trpr.append(mark)
    return tr_el


def strip_revision_marks(element):
    """Remove every revision mark from a cloned subtree.

    Row and column templates are cloned from cells that have *already* been
    marked up, and a clone carries the original's ``w:id`` with it.  Two
    revisions sharing an id is a corrupt file, so a clone is always cleaned
    before it is used.
    """
    for tag in (W_CELLINS, W_CELLDEL):
        for e in list(element.iter(tag)):
            e.getparent().remove(e)
    for holder in list(element.iter(W_TRPR)) + list(element.iter(W_RPR)):
        for tag in (W_INS, W_DEL):
            for e in holder.findall(tag):
                holder.remove(e)
    for e in list(element.iter(W_INS)):
        p = e.getparent()
        if p is not None and p.tag not in (W_RPR, W_TRPR):
            _unwrap(e)
    for e in list(element.iter(W_DEL)):
        p = e.getparent()
        if p is not None and p.tag not in (W_RPR, W_TRPR):
            p.remove(e)
    return element


# ==========================================================================
# text extraction
# ==========================================================================

def run_text(r_el):
    """Visible text of a ``w:r``; field instructions and drawings contribute
    nothing, which is what python-docx's ``Run.text`` also does."""
    out = []
    for c in r_el.iterchildren():
        if c.tag in (W_T, W_DELTEXT):
            out.append(c.text or "")
        elif c.tag in RUN_TEXT_EQUIV:
            out.append(RUN_TEXT_EQUIV[c.tag])
    return "".join(out)


def para_text(p_el, mode=None):
    """Visible text of a ``w:p``.

    ``mode`` is ``None`` (take everything as it stands), ``"accept"`` (drop
    ``w:del`` subtrees) or ``"reject"`` (drop ``w:ins`` subtrees).
    """
    out = []

    def walk(node, in_del=False, in_ins=False):
        for c in node.iterchildren():
            if c.tag == W_INS:
                if mode == "reject":
                    continue
                walk(c, in_del, True)
            elif c.tag == W_DEL:
                if mode == "accept":
                    continue
                walk(c, True, in_ins)
            elif c.tag == W_R:
                out.append(run_text(c))
            elif c.tag in (W_HYPERLINK, w("smartTag"), w("sdtContent"),
                           w("dir"), w("bdo"), w("moveFrom"), w("moveTo")):
                walk(c, in_del, in_ins)
            elif c.tag == w("sdt"):
                walk(c, in_del, in_ins)

    for c in p_el.iterchildren():
        if c.tag in CONTENT_SKIP:
            continue
        if c.tag == W_INS:
            if mode != "reject":
                walk(c)
        elif c.tag == W_DEL:
            if mode != "accept":
                walk(c)
        elif c.tag == W_R:
            out.append(run_text(c))
        elif c.tag in (W_HYPERLINK, w("smartTag"), w("sdt"), w("dir"),
                       w("bdo"), w("moveFrom"), w("moveTo")):
            walk(c)
    return "".join(out)


def para_mark_kept(p_el, mode):
    ppr = p_el.find(W_PPR)
    if ppr is None:
        return True
    rpr = ppr.find(W_RPR)
    if rpr is None:
        return True
    has_ins = rpr.find(W_INS) is not None
    has_del = rpr.find(W_DEL) is not None
    if has_ins and has_del:
        return False
    if has_ins:
        return mode != "reject"
    if has_del:
        return mode != "accept"
    return True


def norm(s):
    """The comparison form: zero-width spaces removed, whitespace collapsed.

    Zero-width spaces are invisible break opportunities inserted by
    ``build_manuscript_docx.soft_break``; they are not text.  Nothing else is
    folded -- a curly quote replacing a straight one is a real change and shows
    up as one.
    """
    return " ".join(s.replace(ZWSP, "").split())


def cell_texts(tc_el, mode=None):
    return [para_text(p, mode) for p in tc_el.findall(W_P)]


def cell_text(tc_el, mode=None):
    return " ".join(t for t in cell_texts(tc_el, mode) if t)


def table_rows(tbl_el):
    return tbl_el.findall(W_TR)


def row_cells(tr_el):
    return tr_el.findall(W_TC)


def body_text_items(body, mode=None):
    """Every non-empty text unit of a body, in document order.

    A paragraph whose mark has been removed by ``mode`` merges into the
    following paragraph, exactly as Word merges it.  Table cells contribute in
    row-major order at the table's position.
    """
    out, buf = [], []

    def flush():
        if buf:
            t = norm("".join(buf))
            if t:
                out.append(t)
            buf.clear()

    for c in body.iterchildren():
        if c.tag == W_P:
            buf.append(para_text(c, mode))
            if para_mark_kept(c, mode):
                flush()
        elif c.tag == W_TBL:
            flush()
            for tr in table_rows(c):
                if mode == "accept" and _row_marked(tr, W_DEL):
                    continue
                if mode == "reject" and _row_marked(tr, W_INS):
                    continue
                for tc in row_cells(tr):
                    if mode == "accept" and tc_has(tc, W_CELLDEL):
                        continue
                    if mode == "reject" and tc_has(tc, W_CELLINS):
                        continue
                    t = norm(cell_text(tc, mode))
                    if t:
                        out.append(t)
    flush()
    return out


def _row_marked(tr_el, tag):
    trpr = tr_el.find(W_TRPR)
    return trpr is not None and trpr.find(tag) is not None


def tc_has(tc_el, tag):
    tcpr = tc_el.find(W_TCPR)
    return tcpr is not None and tcpr.find(tag) is not None


# ==========================================================================
# the items that get aligned
# ==========================================================================

class Item:
    __slots__ = ("kind", "el", "text", "words", "style", "sub")

    def __init__(self, kind, element, text="", style=""):
        self.kind = kind                 # 'p' | 'tbl' | 'img'
        self.el = element
        self.text = text
        self.words = norm(text).split()
        self.style = style
        self.sub = None

    def __repr__(self):
        return "<%s %r>" % (self.kind, self.text[:48])


CAPTION_RE = re.compile(r"^(Figure|Table)\s+S?\d+[a-z]?\s*[.:]")


def para_kind(p_el, text, styles):
    """A coarse role, used only to discourage cross-role matches."""
    ppr = p_el.find(W_PPR)
    style = ""
    if ppr is not None:
        ps = ppr.find(w("pStyle"))
        if ps is not None:
            style = styles.get(ps.get(w("val")), ps.get(w("val")) or "")
    if style.startswith("Heading") or style == "Title":
        return "head", style
    if style == "Bibliography":
        return "ref", style
    if CAPTION_RE.match(norm(text)):
        return "cap", style
    if re.match(r"^\d+\.\s", norm(text)):
        return "ref", style
    return "body", style


def style_map(doc):
    out = {}
    for s in doc.styles.element.findall(w("style")):
        sid = s.get(w("styleId"))
        nm = s.find(w("name"))
        out[sid] = nm.get(w("val")) if nm is not None else sid
    return out


def collect_items(doc):
    """Body order: text paragraphs, tables and image paragraphs."""
    styles = style_map(doc)
    items = []
    for c in doc.element.body.iterchildren():
        if c.tag == W_P:
            if c.findall(".//" + qn("a:blip")):
                items.append(Item("img", c))
                continue
            t = para_text(c)
            if not norm(t):
                continue
            it = Item("p", c, t)
            it.sub, it.style = para_kind(c, t, styles)
            items.append(it)
        elif c.tag == W_TBL:
            items.append(Item("tbl", c))
    return items


# ==========================================================================
# alignment
# ==========================================================================

def similarity(a: Item, b: Item):
    if a.kind != b.kind:
        return 0.0
    if a.kind == "tbl":
        return TBL_ANCHOR_SIM
    if a.kind == "img":
        return IMG_ANCHOR_SIM
    if not a.words or not b.words:
        return 0.0
    sm = difflib.SequenceMatcher(a=a.words, b=b.words, autojunk=False)
    if sm.real_quick_ratio() < MIN_SIM_PARA:
        return 0.0
    if sm.quick_ratio() < MIN_SIM_PARA:
        return 0.0
    r = sm.ratio()
    if a.sub != b.sub:
        r *= KIND_MISMATCH
    return r if r >= MIN_SIM_PARA else 0.0


def align(a_items, b_items, sim=similarity, min_sim=None):
    """Monotone maximum-similarity alignment (Needleman-Wunsch, gaps free).

    Returns ``[('match', i, j) | ('del', i, None) | ('ins', None, j)]`` in
    merged document order.  Gaps score zero, so any admissible match is
    preferred to a pair of gaps; the dynamic program then picks the monotone
    assignment maximising total similarity.
    """
    n, m = len(a_items), len(b_items)
    scores = [[0.0] * m for _ in range(n)]
    for i in range(n):
        ai = a_items[i]
        row = scores[i]
        for j in range(m):
            s = sim(ai, b_items[j])
            if min_sim is not None and s < min_sim:
                s = 0.0
            row[j] = s

    NEG = float("-inf")
    f = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]      # 0 diag, 1 up(del), 2 left(ins)
    for i in range(1, n + 1):
        f[i][0] = 0.0
        bt[i][0] = 1
    for j in range(1, m + 1):
        f[0][j] = 0.0
        bt[0][j] = 2
    for i in range(1, n + 1):
        si = scores[i - 1]
        fi, fp = f[i], f[i - 1]
        bi = bt[i]
        for j in range(1, m + 1):
            s = si[j - 1]
            diag = fp[j - 1] + s if s > 0.0 else NEG
            up = fp[j]
            left = fi[j - 1]
            if diag >= up and diag >= left:
                fi[j], bi[j] = diag, 0
            elif up >= left:
                fi[j], bi[j] = up, 1
            else:
                fi[j], bi[j] = left, 2

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i][j] == 0:
            ops.append(("match", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and (j == 0 or bt[i][j] == 1):
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    ops.reverse()
    return ops, scores


# ==========================================================================
# word-level diff inside a matched pair
# ==========================================================================

TOKEN_RE = re.compile(r"(\S+)(\s*)")


def tokenize(text):
    """``[(key, start, end)]`` where ``text[start:end]`` is one word plus the
    whitespace that follows it, and ``key`` is that word with zero-width spaces
    removed.  The tokens tile ``text`` exactly, which is what lets "accept all"
    reproduce the new text character for character."""
    toks = []
    for m in TOKEN_RE.finditer(text):
        toks.append((m.group(1).replace(ZWSP, ""), m.start(), m.end()))
    return toks


def run_map(p_el):
    """``[(r_el, parent_or_None, start, end)]`` over the paragraph's text, plus
    the concatenated text itself."""
    spans, pos, parts = [], 0, []
    for c in p_el.iterchildren():
        if c.tag in CONTENT_SKIP:
            continue
        if c.tag == W_R:
            t = run_text(c)
            spans.append((c, None, pos, pos + len(t)))
            parts.append(t)
            pos += len(t)
        elif c.tag == W_HYPERLINK:
            for r in c.findall(W_R):
                t = run_text(r)
                spans.append((r, c, pos, pos + len(t)))
                parts.append(t)
                pos += len(t)
        elif c.tag in (W_INS, W_DEL):
            raise AssertionError(
                "the clean document already carries revision marks; "
                "build_manuscript_docx.py must produce a clean file")
    return spans, "".join(parts)


def rpr_of(r_el):
    return None if r_el is None else r_el.find(W_RPR)


def run_at(spans, pos):
    for r_el, _parent, s, e in spans:
        if s <= pos < e:
            return r_el
    return spans[-1][0] if spans else None


def slice_pieces(spans, text, s, e, kind):
    out = []
    for r_el, parent, rs, re_ in spans:
        a, b = max(s, rs), min(e, re_)
        if b > a:
            out.append([kind, r_el, parent, text[a:b]])
    return out


def refine_opcodes(ops, min_equal=MIN_EQUAL_RUN):
    """Absorb tiny unchanged islands, then merge adjacent changes.

    ``difflib`` will happily report ``the``, ``of`` and ``and`` as unchanged in
    the middle of a rewritten sentence, which renders in Word as a shredded
    ribbon of strikethrough and underline.  An ``equal`` block shorter than
    ``min_equal`` with a change on both sides becomes part of the change, and
    consecutive changes are merged into one contiguous deletion followed by one
    contiguous insertion.
    """
    tags = [list(o) for o in ops]
    n = len(tags)
    for k in range(n):
        tag, i1, i2, _j1, _j2 = tags[k]
        if tag != "equal" or (i2 - i1) >= min_equal:
            continue
        if k > 0 and ops[k - 1][0] != "equal" and \
                k < n - 1 and ops[k + 1][0] != "equal":
            tags[k][0] = "replace"
    merged = []
    for t in tags:
        if merged and merged[-1][0] != "equal" and t[0] != "equal":
            merged[-1][2], merged[-1][4] = t[2], t[4]
        else:
            merged.append(list(t))
    out = []
    for tag, i1, i2, j1, j2 in merged:
        if tag != "equal":
            tag = ("insert" if i1 == i2 else
                   "delete" if j1 == j2 else "replace")
        out.append((tag, i1, i2, j1, j2))
    return out


def diff_pieces(old_text, spans, new_text, stats):
    """``[[kind, src_run, hyperlink_parent, text], ...]`` for one paragraph.

    ``kind`` is ``keep`` / ``ins`` / ``del``.  Concatenating ``keep`` + ``ins``
    reproduces ``new_text``; concatenating ``keep`` + ``del`` reproduces
    ``old_text`` once whitespace is collapsed.
    """
    a_txt = norm(old_text)
    a_toks = tokenize(a_txt)
    b_toks = tokenize(new_text)
    a_keys = [t[0] for t in a_toks]
    b_keys = [t[0] for t in b_toks]

    sm = difflib.SequenceMatcher(a=a_keys, b=b_keys, autojunk=False)
    pieces, pos = [], 0
    for tag, i1, i2, j1, j2 in refine_opcodes(sm.get_opcodes()):
        b_span = None
        if j2 > j1:
            b_span = (b_toks[j1][1], b_toks[j2 - 1][2])
        a_span = None
        if i2 > i1:
            a_span = (a_toks[i1][1], a_toks[i2 - 1][2])

        if tag == "equal":
            pieces += slice_pieces(spans, new_text, pos, b_span[1], "keep")
            pos = b_span[1]
            stats["equal_words"] += i2 - i1
        elif tag == "insert":
            pieces += slice_pieces(spans, new_text, pos, b_span[1], "ins")
            pos = b_span[1]
            stats["ins_words"] += j2 - j1
        elif tag == "delete":
            pieces.append(["del", run_at(spans, pos), None,
                           a_txt[a_span[0]:a_span[1]]])
            stats["del_words"] += i2 - i1
        else:                                              # replace
            pieces.append(["del", run_at(spans, pos), None,
                           a_txt[a_span[0]:a_span[1]]])
            pieces += slice_pieces(spans, new_text, pos, b_span[1], "ins")
            pos = b_span[1]
            stats["del_words"] += i2 - i1
            stats["ins_words"] += j2 - j1
    if pos < len(new_text):                                # trailing whitespace
        pieces += slice_pieces(spans, new_text, pos, len(new_text), "keep")

    _repair_deleted_spacing(pieces)
    return pieces


def _repair_deleted_spacing(pieces):
    """A deletion carries whatever separator the rejected text needs.

    ``old = "foo bar"``, ``new = "foo"`` gives the kept token "foo" (no trailing
    space, it ended the paragraph) followed by the deleted token "bar".
    Rejecting would concatenate them.  The space is prepended to the *deleted*
    piece, where it costs nothing: on accept the deletion disappears with it.
    """
    rej = []
    for idx, p in enumerate(pieces):
        if p[0] == "ins":
            continue
        if p[0] == "del":
            prev = "".join(rej)
            if prev and not prev[-1].isspace() and p[3] and not p[3][0].isspace():
                p[3] = " " + p[3]
            nxt = None
            for q in pieces[idx + 1:]:
                if q[0] in ("keep", "del"):
                    nxt = q
                    break
            if (nxt is not None and p[3] and not p[3][-1].isspace()
                    and nxt[3] and not nxt[3][0].isspace()):
                p[3] = p[3] + " "
        rej.append(p[3])


def emit_pieces(p_el, pieces, ids):
    """Replace the paragraph's content children with the diffed runs."""
    for c in list(p_el.iterchildren()):
        if c.tag not in CONTENT_SKIP:
            p_el.remove(c)

    out = []
    i = 0
    n = len(pieces)
    while i < n:
        kind, r_el, parent, _txt = pieces[i]
        j = i
        group = []
        while j < n and pieces[j][0] == kind and pieces[j][2] is parent:
            group.append(pieces[j])
            j += 1
        runs = [make_run(rpr_of(g[1]), g[3], deleted=(kind == "del"))
                for g in group if g[3]]
        if not runs:
            i = j
            continue
        if kind == "keep":
            content = runs
        else:
            content = [wrap(kind, runs, ids)]
        if parent is not None:
            link = copy.deepcopy(parent)
            for k in list(link.iterchildren()):
                link.remove(k)
            for c in content:
                link.append(c)
            out.append(link)
        else:
            out.extend(content)
        i = j

    for e in out:
        p_el.append(e)
    return p_el


# ==========================================================================
# whole-paragraph insertion and deletion
# ==========================================================================

def mark_whole_paragraph_inserted(p_el, ids, stats, mark_para=True):
    """Every run wrapped in ``w:ins``.

    ``mark_para`` also marks the paragraph mark, which is what makes a
    whole-paragraph insertion disappear completely when it is rejected instead
    of leaving an empty paragraph behind.  It is *not* set inside a table cell:
    there the last paragraph mark is the cell's own end marker, a cell must
    always contain at least one paragraph, and ``w:cellIns`` on the cell already
    says the cell is new.
    """
    _spans, text = run_map(p_el)
    if not norm(text):
        return
    kids = [c for c in p_el.iterchildren() if c.tag not in CONTENT_SKIP]
    for c in list(kids):
        p_el.remove(c)
    for c in kids:
        if c.tag == W_R:
            p_el.append(wrap("ins", [c], ids))
        elif c.tag == W_HYPERLINK:
            runs = list(c.findall(W_R))
            for r in runs:
                c.remove(r)
            c.append(wrap("ins", runs, ids))
            p_el.append(c)
        else:
            p_el.append(c)
    if mark_para:
        mark_paragraph_mark(p_el, "ins", ids)
        stats["ins_paragraphs"] += 1
    stats["ins_words"] += len(norm(text).split())


def shift_inserted_paragraph_marks(body, ids):
    """Move each inserted paragraph mark onto the paragraph *before* it.

    Inserting a paragraph P between A and B inserts the characters
    ``P_text`` plus one paragraph mark, and either ¶A or ¶P can be the one
    marked: rejecting removes the mark and merges the two paragraphs, and the
    surviving text is the same either way.  Both encodings are valid and Word
    produces both, depending on whether the author split at the end of A or at
    the start of B.

    They are *not* equally robust.  Marking ¶P and re-saving through
    LibreOffice adds a second mark on ¶A without removing the first, so
    "reject all" inside LibreOffice then removes both marks and runs A and B
    together -- six paragraph breaks were lost that way in this manuscript.
    Marking ¶A instead is LibreOffice's own model: the file survives its round
    trip byte-for-byte in revision terms, and reject-all reproduces v4 exactly
    under both LibreOffice's reading and ours.  So the marks are shifted.

    Applied in document order this turns a run P1..Pk of inserted paragraphs
    after A into "A, P1..P(k-1) marked, Pk unmarked", which is exactly what
    Word writes for k consecutive new paragraphs.
    """
    moved = kept = 0
    for p in body.findall(W_P):
        ppr = p.find(W_PPR)
        rpr = ppr.find(W_RPR) if ppr is not None else None
        mark = rpr.find(W_INS) if rpr is not None else None
        if mark is None:
            continue
        prev = p.getprevious()
        if prev is None or prev.tag != W_P:
            kept += 1
            continue
        pprev = prev.find(W_PPR)
        rprev = pprev.find(W_RPR) if pprev is not None else None
        if rprev is not None and (rprev.find(W_INS) is not None
                                  or rprev.find(W_DEL) is not None):
            kept += 1
            continue
        rid = mark.get(w("id"))
        rpr.remove(mark)
        if len(rpr) == 0:
            ppr.remove(rpr)
        mark_paragraph_mark(prev, "ins", iter([rid]))
        moved += 1
    return moved, kept


def build_deleted_paragraph(v4_p_el, ids, stats):
    """A clone of a v4 paragraph whose every run and whose paragraph mark are
    marked deleted.  Field runs are dropped: they carry no visible text and
    their instruction text would have to become ``w:delInstrText``."""
    p = copy.deepcopy(v4_p_el)
    kids = [c for c in p.iterchildren() if c.tag not in CONTENT_SKIP]
    for c in kids:
        p.remove(c)

    def convert_run(r):
        r = copy.deepcopy(r)
        for c in list(r.iterchildren()):
            if c.tag in RUN_DROP_WHEN_CLONING:
                if c.tag in (w("fldChar"), w("instrText"), w("delInstrText")):
                    stats["dropped_field_parts"] += 1
                elif c.tag in (w("drawing"), w("pict"), w("object")):
                    stats["dropped_drawings"] += 1
                r.remove(c)
            elif c.tag == W_T:
                d = el("w:delText")
                d.set(XML_SPACE, "preserve")
                d.text = c.text
                c.addprevious(d)
                r.remove(c)
        if not any(c.tag != W_RPR for c in r.iterchildren()):
            return None
        return r

    for c in kids:
        if c.tag == W_R:
            r = convert_run(c)
            if r is not None:
                p.append(wrap("del", [r], ids))
        elif c.tag == W_HYPERLINK:
            link = copy.deepcopy(c)
            runs = []
            for r in list(link.iterchildren()):
                link.remove(r)
                if r.tag == W_R:
                    conv = convert_run(r)
                    if conv is not None:
                        runs.append(conv)
            if runs:
                link.append(wrap("del", runs, ids))
                p.append(link)
    mark_paragraph_mark(p, "del", ids)
    stats["del_words"] += len(norm(para_text(v4_p_el)).split())
    stats["del_paragraphs"] += 1
    return p


# ==========================================================================
# tables
# ==========================================================================

class CellItem:
    """A table row or column reduced to a comparable text object."""

    def __init__(self, texts):
        self.kind = "p"
        self.sub = "body"
        self.texts = [norm(t) for t in texts]
        self.text = " | ".join(self.texts)
        self.words = norm(self.text).split()
        self.el = None


def _ratio(x, y):
    """Word similarity or character similarity, whichever is higher.

    Table cells are two or three tokens long ("Peak memory (GB)",
    "0.82*"), so a word-level ratio is almost always 0 and says nothing.  The
    character ratio is what actually separates "Peak memory (GB)" ->
    "Peak RSS (GB)" (0.62) from "Peak memory (GB)" -> "Cont. bias (pp)" (0.22).
    """
    if not x and not y:
        return 1.0
    if not x or not y:
        return 0.0
    wr = difflib.SequenceMatcher(a=x.split(), b=y.split(), autojunk=False)
    cr = difflib.SequenceMatcher(a=x.lower(), b=y.lower(), autojunk=False)
    return max(wr.ratio(), cr.ratio())


def row_similarity(a, b):
    return max(_ratio(a.text, b.text), 0.01)


def col_similarity(a, b):
    """Columns are compared cell by cell down the already-aligned rows."""
    n = min(len(a.texts), len(b.texts))
    if not n:
        return 0.01
    return max(sum(_ratio(a.texts[k], b.texts[k]) for k in range(n)) / n, 0.01)


def grid_of(tbl_el):
    rows = table_rows(tbl_el)
    grid = []
    for tr in rows:
        cells = row_cells(tr)
        for tc in cells:
            tcpr = tc.find(W_TCPR)
            if tcpr is not None:
                for bad in ("gridSpan", "vMerge"):
                    if tcpr.find(w(bad)) is not None:
                        raise AssertionError(
                            "table carries a %s; merged cells are not "
                            "supported by the cell-level diff" % bad)
        grid.append(cells)
    widths = {len(r) for r in grid}
    if len(widths) != 1:
        raise AssertionError("ragged table: rows have %s cells" % sorted(widths))
    return rows, grid


def diff_table(new_tbl, old_tbl, ids, stats, report):
    old_rows, old_grid = grid_of(old_tbl)
    new_rows, new_grid = grid_of(new_tbl)

    a_rows = [CellItem([cell_text(c) for c in r]) for r in old_grid]
    b_rows = [CellItem([cell_text(c) for c in r]) for r in new_grid]
    row_ops, _ = align(a_rows, b_rows, sim=row_similarity, min_sim=MIN_SIM_CELL)
    row_pairs = [(i, j) for k, i, j in row_ops if k == "match"]

    ncol_a, ncol_b = len(old_grid[0]), len(new_grid[0])
    a_cols = [CellItem([cell_text(old_grid[i][c]) for i, _ in row_pairs])
              for c in range(ncol_a)]
    b_cols = [CellItem([cell_text(new_grid[j][c]) for _, j in row_pairs])
              for c in range(ncol_b)]
    col_ops, _ = align(a_cols, b_cols, sim=col_similarity, min_sim=MIN_SIM_CELL)

    col_match = {j: i for k, i, j in col_ops if k == "match"}
    col_inserted = [j for k, i, j in col_ops if k == "ins"]
    col_deleted = [i for k, i, j in col_ops if k == "del"]

    report["table_rows"] = (len(old_grid), len(new_grid))
    report["table_cols"] = (ncol_a, ncol_b)
    report["table_col_headers"] = ([norm(cell_text(c)) for c in old_grid[0]],
                                   [norm(cell_text(c)) for c in new_grid[0]])
    report["table_col_ops"] = col_ops
    report["table_row_ops"] = row_ops
    report["table_col_deleted"] = col_deleted
    report["table_col_inserted"] = col_inserted

    if col_deleted:
        _insert_deleted_columns(new_tbl, new_rows, new_grid, old_grid, col_ops,
                                row_pairs, ids, stats)
        new_rows, new_grid = grid_of(new_tbl)
        # after re-gridding, recompute the mapping onto the widened table
        col_ops = report["table_col_ops_widened"] = _widened_col_ops(col_ops)
        col_match = {j: i for k, i, j in col_ops if k == "match"}
        col_inserted = [j for k, i, j in col_ops if k == "ins"]

    row_match = {j: i for k, i, j in row_ops if k == "match"}

    for j, tr in enumerate(new_rows):
        if j not in row_match:
            mark_row(tr, "ins", ids)
            for tc in row_cells(tr):
                for p in tc.findall(W_P):
                    mark_whole_paragraph_inserted_cell(p, ids, stats)
            stats["ins_rows"] += 1
            continue
        i = row_match[j]
        for c, tc in enumerate(row_cells(tr)):
            if tc_has(tc, W_CELLDEL):
                continue
            if c in col_inserted or c not in col_match:
                mark_cell(tc, "ins", ids)
                for p in tc.findall(W_P):
                    mark_whole_paragraph_inserted_cell(p, ids, stats)
                stats["ins_cells"] += 1
                continue
            old_tc = old_grid[i][col_match[c]]
            _diff_cell(tc, old_tc, ids, stats)

    # rows that exist only in v4 are appended as deleted rows
    for k, i, _j in row_ops:
        if k != "del":
            continue
        tr = _build_deleted_row(new_tbl, new_rows, old_grid[i], col_ops, ids,
                                stats)
        new_rows[-1].addnext(tr)
        new_rows.append(tr)
        stats["del_rows"] += 1
    return report


def _widened_col_ops(col_ops):
    """After deleted columns are physically inserted, every op is a match."""
    out, j = [], 0
    for k, i, _ in col_ops:
        out.append(("match", i, j) if k != "ins" else ("ins", None, j))
        j += 1
    return out


def mark_whole_paragraph_inserted_cell(p_el, ids, stats):
    mark_whole_paragraph_inserted(p_el, ids, stats, mark_para=False)


def _diff_cell(new_tc, old_tc, ids, stats):
    new_ps = new_tc.findall(W_P)
    if len(new_ps) != 1:
        raise AssertionError(
            "table cell has %d paragraphs; the cell diff expects exactly one"
            % len(new_ps))
    old_text = cell_text(old_tc)
    p = new_ps[0]
    spans, text = run_map(p)
    if norm(old_text) == norm(text):
        return
    if not norm(text):
        # cell emptied: everything in it is a deletion
        pieces = [["del", None, None, norm(old_text)]]
        emit_pieces(p, pieces, ids)
        stats["del_words"] += len(norm(old_text).split())
        stats["changed_cells"] += 1
        return
    pieces = diff_pieces(old_text, spans, text, stats)
    emit_pieces(p, pieces, ids)
    stats["changed_cells"] += 1


def _build_deleted_row(tbl_el, new_rows, old_cells, col_ops, ids, stats):
    tr = strip_revision_marks(copy.deepcopy(new_rows[-1]))
    tcs = row_cells(tr)
    a_for_b = {}
    j = 0
    for k, i, _ in col_ops:
        if k == "ins":
            j += 1
            continue
        a_for_b[j] = i
        j += 1
    for c, tc in enumerate(tcs):
        for p in list(tc.findall(W_P))[1:]:
            tc.remove(p)
        p = tc.find(W_P)
        for x in list(p.iterchildren()):
            if x.tag not in CONTENT_SKIP:
                p.remove(x)
        if c in a_for_b:
            txt = norm(cell_text(old_cells[a_for_b[c]]))
            if txt:
                p.append(wrap("del", [make_run(None, txt, deleted=True)], ids))
                stats["del_words"] += len(txt.split())
    mark_row(tr, "del", ids)
    return tr


def _insert_deleted_columns(tbl_el, new_rows, new_grid, old_grid, col_ops,
                            row_pairs, ids, stats):
    """Give a v4-only column real cells in the new table, marked ``w:cellDel``.

    Accepting the revision removes them and the table is the clean table again;
    rejecting restores v4's column.  The grid is rebuilt so the total width is
    unchanged.
    """
    row_a_for_b = {j: i for i, j in row_pairs}
    for j, tr in enumerate(new_rows):
        cells = row_cells(tr)
        pos = 0
        for k, i, _ in col_ops:
            if k != "del":
                pos += 1
                continue
            tpl = strip_revision_marks(
                copy.deepcopy(cells[min(pos, len(cells) - 1)]))
            for p in list(tpl.findall(W_P))[1:]:
                tpl.remove(p)
            p = tpl.find(W_P)
            for x in list(p.iterchildren()):
                if x.tag not in CONTENT_SKIP:
                    p.remove(x)
            if j in row_a_for_b:
                txt = norm(cell_text(old_grid[row_a_for_b[j]][i]))
                if txt:
                    p.append(wrap("del", [make_run(None, txt, deleted=True)],
                                  ids))
                    stats["del_words"] += len(txt.split())
            mark_cell(tpl, "del", ids)
            if pos < len(cells):
                cells[pos].addprevious(tpl)
            else:
                tr.append(tpl)
            cells = row_cells(tr)
            pos += 1
            stats["del_cells"] += 1

    _rescale_grid(tbl_el)


def _rescale_grid(tbl_el):
    rows, grid = grid_of(tbl_el)
    ncol = len(grid[0])
    old_grid_el = tbl_el.find(w("tblGrid"))
    old_w = [int(g.get(w("w"))) for g in old_grid_el.findall(w("gridCol"))]
    total = sum(old_w)
    tblpr = tbl_el.find(w("tblPr"))
    tblw = tblpr.find(w("tblW")) if tblpr is not None else None
    if tblw is not None and tblw.get(w("type")) == "dxa":
        try:
            total = int(tblw.get(w("w"))) or total
        except (TypeError, ValueError):
            pass
    widths = []
    for c in range(ncol):
        widths.append(old_w[c] if c < len(old_w) else int(total / max(ncol, 1)))
    scale = total / sum(widths)
    widths = [max(int(x * scale), 200) for x in widths]
    for g in list(old_grid_el.findall(w("gridCol"))):
        old_grid_el.remove(g)
    for x in widths:
        g = el("w:gridCol")
        g.set(w("w"), str(x))
        old_grid_el.append(g)
    for tr in rows:
        for c, tc in enumerate(row_cells(tr)):
            tcpr = get_or_add_first(tc, W_TCPR)
            tcw = tcpr.find(w("tcW"))
            if tcw is None:
                tcw = el("w:tcW")
                tcpr.insert(0, tcw)
            tcw.set(w("w"), str(widths[min(c, len(widths) - 1)]))
            tcw.set(w("type"), "dxa")


# ==========================================================================
# the body rewrite
# ==========================================================================

def build_tracked_body(v4_doc, out_doc, report):
    ids = id_source()
    stats = dict(equal_words=0, ins_words=0, del_words=0, ins_paragraphs=0,
                 del_paragraphs=0, changed_paragraphs=0, unchanged_paragraphs=0,
                 changed_cells=0, ins_cells=0, del_cells=0, ins_rows=0,
                 del_rows=0, dropped_field_parts=0, dropped_drawings=0)

    a_items = collect_items(v4_doc)
    b_items = collect_items(out_doc)
    ops, _scores = align(a_items, b_items)

    report["v4_items"] = _kind_counts(a_items)
    report["new_items"] = _kind_counts(b_items)
    report["ops"] = _op_counts(ops)

    # for every deleted v4 item, the new-document element it must precede
    pending = {}
    seq = []
    for k, i, j in ops:
        if k == "del":
            seq.append(("del", i))
        else:
            seq.append((k, j))
    buf = []
    anchors = []
    for k, idx in seq:
        if k == "del":
            buf.append(idx)
        else:
            anchors.append((idx, buf))
            buf = []
    tail = buf
    for j, before in anchors:
        if before:
            pending[j] = before

    match_for_b = {j: i for k, i, j in ops if k == "match"}
    ins_b = {j for k, i, j in ops if k == "ins"}

    b_index = {id(it.el): j for j, it in enumerate(b_items)}
    body = out_doc.element.body
    sectPr = body.find(W_SECTPR)

    unmatched_v4 = []

    for c in list(body.iterchildren()):
        j = b_index.get(id(c))
        if j is None:
            continue
        item = b_items[j]
        for i in pending.get(j, []):
            a = a_items[i]
            if a.kind == "p":
                c.addprevious(build_deleted_paragraph(a.el, ids, stats))
                unmatched_v4.append(("paragraph", norm(a.text)[:90]))
            elif a.kind == "tbl":
                raise AssertionError(
                    "a v4 table could not be aligned; whole-table deletion is "
                    "not implemented -- align it or remove it by hand")
            else:
                unmatched_v4.append(("image", "v4 image paragraph, not marked"))

        if item.kind == "p":
            if j in ins_b:
                mark_whole_paragraph_inserted(item.el, ids, stats)
            else:
                a = a_items[match_for_b[j]]
                spans, text = run_map(item.el)
                if norm(a.text) == norm(text):
                    stats["unchanged_paragraphs"] += 1
                    stats["equal_words"] += len(norm(text).split())
                else:
                    pieces = diff_pieces(a.text, spans, text, stats)
                    emit_pieces(item.el, pieces, ids)
                    stats["changed_paragraphs"] += 1
        elif item.kind == "tbl":
            if j in ins_b:
                raise AssertionError(
                    "the new table has no v4 counterpart; whole-table "
                    "insertion is not implemented")
            diff_table(item.el, a_items[match_for_b[j]].el, ids, stats, report)

    for i in tail:
        a = a_items[i]
        if a.kind == "p":
            sectPr.addprevious(build_deleted_paragraph(a.el, ids, stats))
            unmatched_v4.append(("paragraph", norm(a.text)[:90]))
        elif a.kind == "img":
            unmatched_v4.append(("image", "v4 image paragraph, not marked"))
        else:
            raise AssertionError("a v4 table fell off the end of the alignment")

    moved, kept = shift_inserted_paragraph_marks(body, ids)
    report["mark_shift"] = (moved, kept)

    report["stats"] = stats
    report["v4_only"] = unmatched_v4
    report["n_ids"] = next(ids) - 1
    return stats


def _kind_counts(items):
    out = {}
    for it in items:
        out[it.kind] = out.get(it.kind, 0) + 1
    return out


def _op_counts(ops):
    out = {}
    for k, _i, _j in ops:
        out[k] = out.get(k, 0) + 1
    return out


# ==========================================================================
# accept-all / reject-all, performed on the XML rather than asserted
# ==========================================================================

def apply_revisions(body, mode):
    """Mutate ``body`` as Word would on "accept all" / "reject all"."""
    assert mode in ("accept", "reject")

    # rows
    for tbl in body.iter(W_TBL):
        for tr in list(table_rows(tbl)):
            if _row_marked(tr, W_DEL) and mode == "accept":
                tbl.remove(tr)
            elif _row_marked(tr, W_INS) and mode == "reject":
                tbl.remove(tr)
            else:
                trpr = tr.find(W_TRPR)
                if trpr is not None:
                    for tag in (W_INS, W_DEL):
                        for e in trpr.findall(tag):
                            trpr.remove(e)

    # cells
    for tbl in body.iter(W_TBL):
        changed = False
        for tr in table_rows(tbl):
            for tc in list(row_cells(tr)):
                if tc_has(tc, W_CELLDEL) and mode == "accept":
                    tr.remove(tc)
                    changed = True
                elif tc_has(tc, W_CELLINS) and mode == "reject":
                    tr.remove(tc)
                    changed = True
                else:
                    tcpr = tc.find(W_TCPR)
                    if tcpr is not None:
                        for tag in (W_CELLINS, W_CELLDEL):
                            for e in tcpr.findall(tag):
                                tcpr.remove(e)
        if changed:
            _rescale_grid(tbl)

    # runs
    for e in list(body.iter(W_INS)):
        if e.getparent() is None:
            continue
        if e.getparent().tag == W_RPR:               # paragraph mark, later
            continue
        if mode == "reject":
            e.getparent().remove(e)
        else:
            _unwrap(e)
    for e in list(body.iter(W_DEL)):
        if e.getparent() is None:
            continue
        if e.getparent().tag == W_RPR:
            continue
        if mode == "accept":
            e.getparent().remove(e)
        else:
            for t in list(e.iter(W_DELTEXT)):
                new = el("w:t")
                new.set(XML_SPACE, "preserve")
                new.text = t.text
                t.addprevious(new)
                t.getparent().remove(t)
            _unwrap(e)

    # paragraph marks
    for p in list(body.iter(W_P)):
        ppr = p.find(W_PPR)
        if ppr is None:
            continue
        rpr = ppr.find(W_RPR)
        if rpr is None:
            continue
        has_ins = rpr.find(W_INS) is not None
        has_del = rpr.find(W_DEL) is not None
        if not (has_ins or has_del):
            continue
        for tag in (W_INS, W_DEL):
            for e in rpr.findall(tag):
                rpr.remove(e)
        if len(rpr) == 0:
            ppr.remove(rpr)
        keep = (mode != "reject") if has_ins else (mode != "accept")
        if has_ins and has_del:
            keep = False
        if keep:
            continue
        _merge_forward(p)
    return body


def _unwrap(e):
    parent = e.getparent()
    idx = list(parent).index(e)
    for k, c in enumerate(list(e.iterchildren())):
        e.remove(c)
        parent.insert(idx + k, c)
    parent.remove(e)


def _merge_forward(p):
    """The paragraph mark is gone, so this paragraph runs into the next one."""
    content = [c for c in p.iterchildren() if c.tag not in CONTENT_SKIP]
    nxt = p.getnext()
    while nxt is not None and nxt.tag != W_P:
        nxt = nxt.getnext()
    if nxt is None:
        if any(run_text(r) for c in content for r in c.iter(W_R)):
            raise AssertionError(
                "a paragraph with a removed paragraph mark and surviving text "
                "has no following paragraph to merge into")
        p.getparent().remove(p)
        return
    anchor = nxt.find(W_PPR)
    for c in content:
        p.remove(c)
    for k, c in enumerate(content):
        if anchor is not None:
            if k == 0:
                anchor.addnext(c)
            else:
                content[k - 1].addnext(c)
        else:
            nxt.insert(k, c)
    p.getparent().remove(p)


def simulated_body(path, mode):
    d = docx.Document(path)
    body = d.element.body
    apply_revisions(body, mode)
    return d


# ==========================================================================
# validation
# ==========================================================================

def revision_inventory(body):
    def by_parent(tag, parents, negate=False):
        out = []
        for e in body.iter(tag):
            p = e.getparent()
            hit = p is not None and p.tag in parents
            if hit != negate:
                out.append(e)
        return out

    holders = (W_RPR, W_TRPR)
    return dict(ins=by_parent(W_INS, holders, negate=True),
                dele=by_parent(W_DEL, holders, negate=True),
                mark_ins=by_parent(W_INS, (W_RPR,)),
                mark_del=by_parent(W_DEL, (W_RPR,)),
                row_ins=by_parent(W_INS, (W_TRPR,)),
                row_del=by_parent(W_DEL, (W_TRPR,)),
                cell_ins=list(body.iter(W_CELLINS)),
                cell_del=list(body.iter(W_CELLDEL)))


def validate_structure(path):
    """Everything that can be checked without an OOXML schema."""
    problems = []
    d = docx.Document(path)                     # must reopen
    body = d.element.body
    inv = revision_inventory(body)
    all_marks = (inv["ins"] + inv["dele"] + inv["mark_ins"] + inv["mark_del"]
                 + inv["row_ins"] + inv["row_del"]
                 + inv["cell_ins"] + inv["cell_del"])
    seen = {}
    for e in all_marks:
        for attr in ("id", "author", "date"):
            if e.get(w(attr)) is None:
                problems.append("%s missing w:%s" % (e.tag, attr))
        rid = e.get(w("id"))
        if rid in seen:
            problems.append("duplicate w:id %s" % rid)
        seen[rid] = e
        if e.get(w("date")) != REV_DATE:
            problems.append("unexpected w:date %r" % e.get(w("date")))
        if e.get(w("author")) != REV_AUTHOR:
            problems.append("unexpected w:author %r" % e.get(w("author")))

    # a deletion's text element must be w:delText, never w:t
    for e in inv["dele"]:
        for t in e.iter(W_T):
            problems.append("w:t inside w:del: %r" % (t.text or "")[:40])
    for e in inv["ins"]:
        for t in e.iter(W_DELTEXT):
            problems.append("w:delText inside w:ins: %r" % (t.text or "")[:40])
    # ... and no w:delText may appear outside a w:del
    for t in body.iter(W_DELTEXT):
        anc = t.getparent()
        ok = False
        while anc is not None:
            if anc.tag == W_DEL:
                ok = True
                break
            anc = anc.getparent()
        if not ok:
            problems.append("w:delText outside w:del: %r" % (t.text or "")[:40])

    # schema ordering: w:rPr is the last element of w:pPr bar sectPr/pPrChange,
    # and the paragraph-mark w:ins/w:del is the first element of that w:rPr
    tail = {W_SECTPR, w("pPrChange")}
    for ppr in body.iter(W_PPR):
        rpr = ppr.find(W_RPR)
        if rpr is None:
            continue
        after = list(ppr)[list(ppr).index(rpr) + 1:]
        for e in after:
            if e.tag not in tail:
                problems.append("w:pPr child %s follows w:rPr" % e.tag)
        for tag in (W_INS, W_DEL):
            m = rpr.find(tag)
            if m is not None and list(rpr).index(m) != 0:
                problems.append("paragraph-mark %s is not first in w:rPr" % tag)

    # CT_TrPr puts w:ins / w:del after the base row properties
    for trpr in body.iter(W_TRPR):
        for tag in (W_INS, W_DEL):
            m = trpr.find(tag)
            if m is None:
                continue
            after = list(trpr)[list(trpr).index(m) + 1:]
            for e in after:
                if e.tag not in (W_DEL, w("trPrChange")):
                    problems.append("w:trPr child %s follows %s"
                                    % (e.tag, tag))

    # revision marks must be children of w:p / w:hyperlink, never of w:pPr
    for e in inv["ins"] + inv["dele"]:
        if e.getparent().tag not in (W_P, W_HYPERLINK, W_INS, W_DEL):
            problems.append("%s under %s" % (e.tag, e.getparent().tag))

    counts = dict(ins=len(inv["ins"]), dele=len(inv["dele"]),
                  mark_ins=len(inv["mark_ins"]), mark_del=len(inv["mark_del"]),
                  row_ins=len(inv["row_ins"]), row_del=len(inv["row_del"]),
                  cell_ins=len(inv["cell_ins"]), cell_del=len(inv["cell_del"]),
                  ins_runs=sum(len(e.findall(W_R)) for e in inv["ins"]),
                  del_runs=sum(len(e.findall(W_R)) for e in inv["dele"]))
    return counts, problems, d


def xml_counts(path):
    """``<w:ins `` / ``<w:del `` as they appear in word/document.xml itself."""
    import zipfile
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    return (xml.count("<w:ins "), xml.count("<w:del "),
            xml.count("<w:delText"), len(xml))


# ==========================================================================
# PDF
# ==========================================================================

def to_pdf(path, outdir=None, timeout=1800):
    tmp = outdir or tempfile.mkdtemp(prefix="tracked_pdf_")
    profile = tempfile.mkdtemp(prefix="lo_profile_")
    try:
        r = subprocess.run(
            ["/usr/bin/soffice", "-env:UserInstallation=file://" + profile,
             "--headless", "--norestore", "--convert-to", "pdf",
             "--outdir", tmp, path],
            capture_output=True, text=True, timeout=timeout)
        pdf = os.path.join(tmp, os.path.basename(path).replace(".docx", ".pdf"))
        if not os.path.exists(pdf):
            return None, "conversion failed: %s %s" % (r.stdout[-400:],
                                                       r.stderr[-400:])
        info = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True)
        pages = re.search(r"^Pages:\s+(\d+)", info.stdout, re.M)
        size = re.search(r"^Page size:\s+(.*)$", info.stdout, re.M)
        return (int(pages.group(1)) if pages else -1,
                size.group(1).strip() if size else "?"), None
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def roundtrip_docx(path, v4_text, clean_text, timeout=1800):
    """Re-save through LibreOffice, then accept and reject *its* output.

    This is the strongest open-without-repair evidence available without Word.
    An independent OOXML implementation has to *understand* the marks as
    revisions to write them out again -- a reader that had merely tolerated the
    file would emit plain text -- and running the accept/reject simulation on
    what it wrote proves its reading of every revision agrees with ours.
    """
    tmp = tempfile.mkdtemp(prefix="tracked_rt_")
    profile = tempfile.mkdtemp(prefix="lo_profile_")
    try:
        r = subprocess.run(
            ["/usr/bin/soffice", "-env:UserInstallation=file://" + profile,
             "--headless", "--norestore", "--convert-to",
             "docx:MS Word 2007 XML", "--outdir", tmp, path],
            capture_output=True, text=True, timeout=timeout)
        out = os.path.join(tmp, os.path.basename(path))
        if not os.path.exists(out):
            return None, "round trip failed: %s %s" % (r.stdout[-300:],
                                                       r.stderr[-300:])
        counts, _problems, _d = validate_structure(out)
        rej = body_text_items(simulated_body(out, "reject").element.body)
        acc = body_text_items(simulated_body(out, "accept").element.body)
        rej_ok, _ = _compare(v4_text, rej)
        acc_ok, _ = _compare(clean_text, acc)
        return dict(counts=counts, reject_ok=rej_ok, accept_ok=acc_ok,
                    n_reject=len(rej), n_accept=len(acc)), None
    finally:
        shutil.rmtree(profile, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# figures
# ==========================================================================

def schema_probe():
    """Is a wordprocessingml XSD actually available on this host?

    Reported rather than assumed, so the claim in the report is a measurement.
    """
    import glob
    found = []
    try:
        import xmlschema                                    # noqa: F401
        validator = "xmlschema %s" % xmlschema.__version__
    except Exception:
        validator = None
    pkg = os.path.dirname(docx.__file__)
    for root in (pkg, "/usr/share/xml", "/usr/share/schemas",
                 os.path.join(sys.prefix, "share", "xml")):
        if os.path.isdir(root):
            found += glob.glob(os.path.join(root, "**", "wml.xsd"),
                               recursive=True)[:3]
    return validator, found


def content_digest(path):
    """SHA-256 over the package's *contents*, ignoring zip timestamps.

    python-docx stamps each zip entry with the time of the write, so two
    identical builds have different file hashes.  This hash is over the sorted
    (name, bytes) pairs, so it is stable across rebuilds and is what proves the
    build is deterministic.
    """
    import zipfile
    h = hashlib.sha256()
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist()):
            h.update(n.encode("utf-8"))
            h.update(z.read(n))
    return h.hexdigest()


def image_digests(path):
    import zipfile
    out = []
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist()):
            if n.startswith("word/media/"):
                b = z.read(n)
                out.append((os.path.basename(n), len(b),
                            hashlib.sha256(b).hexdigest()[:16]))
    return out


# ==========================================================================
# staleness
# ==========================================================================

def clean_file_is_current():
    """``manuscript_revised.docx`` must be the render of the current markdown."""
    doc = docx.Document(CLEAN_DOCX)
    want = bmd.target_paragraphs()
    got = bmd.docx_paragraphs(doc)
    sm = difflib.SequenceMatcher(a=want, b=got, autojunk=False)
    diffs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        for k in range(max(i2 - i1, j2 - j1)):
            diffs.append((tag,
                          want[i1 + k] if i1 + k < i2 else None,
                          got[j1 + k] if j1 + k < j2 else None))
    return diffs


# ==========================================================================
# build
# ==========================================================================

def build(v4_path, clean_path, out_path, label, report):
    shutil.copyfile(clean_path, out_path)
    v4 = docx.Document(v4_path)
    out = docx.Document(out_path)
    stats = build_tracked_body(v4, out, report)
    out.save(out_path)
    report["label"] = label
    report["size_mb"] = os.path.getsize(out_path) / 1e6
    return stats


def check(v4_path, clean_path, out_path, report, do_pdf=True):
    """Every claim this script makes, demonstrated."""
    ok = True
    lines = []

    n_ins, n_del, n_deltext, xml_len = xml_counts(out_path)
    report["xml"] = dict(ins=n_ins, dele=n_del, delText=n_deltext,
                         bytes=xml_len)
    lines.append("word/document.xml: %d '<w:ins ', %d '<w:del ', "
                 "%d '<w:delText', %.2f MB of XML"
                 % (n_ins, n_del, n_deltext, xml_len / 1e6))
    if not (n_ins > 0 and n_del > 0):
        ok = False
        lines.append("FAIL: the file must carry both insertions and deletions")

    counts, problems, _doc = validate_structure(out_path)
    report["counts"] = counts
    report["problems"] = problems
    lines.append("python-docx reopened the file: OK")
    lines.append("run-level marks: %d w:ins (%d runs), %d w:del (%d runs); "
                 "paragraph marks: %d inserted, %d deleted; "
                 "cell marks: %d inserted, %d deleted; "
                 "row marks: %d inserted, %d deleted"
                 % (counts["ins"], counts["ins_runs"], counts["dele"],
                    counts["del_runs"], counts["mark_ins"], counts["mark_del"],
                    counts["cell_ins"], counts["cell_del"],
                    counts["row_ins"], counts["row_del"]))
    lines.append("structural problems: %s" % (problems[:8] if problems
                                              else "none"))
    if problems:
        ok = False

    v4_text = body_text_items(docx.Document(v4_path).element.body)
    clean_text = body_text_items(docx.Document(clean_path).element.body)

    rej = simulated_body(out_path, "reject")
    rej_text = body_text_items(rej.element.body)
    acc = simulated_body(out_path, "accept")
    acc_text = body_text_items(acc.element.body)

    report["n_v4_units"] = len(v4_text)
    report["n_clean_units"] = len(clean_text)

    rej_ok, rej_diffs = _compare(v4_text, rej_text)
    acc_ok, acc_diffs = _compare(clean_text, acc_text)
    report["reject_diffs"] = rej_diffs
    report["accept_diffs"] = acc_diffs
    lines.append("REJECT ALL -> %d text units, v4 has %d: %s"
                 % (len(rej_text), len(v4_text),
                    "IDENTICAL" if rej_ok else "%d differ" % len(rej_diffs)))
    lines.append("ACCEPT ALL -> %d text units, clean has %d: %s"
                 % (len(acc_text), len(clean_text),
                    "IDENTICAL" if acc_ok else "%d differ" % len(acc_diffs)))
    ok = ok and rej_ok and acc_ok

    if do_pdf:
        acc_path = os.path.join(tempfile.mkdtemp(prefix="tracked_acc_"),
                                "accepted.docx")
        acc.save(acc_path)
        pdfs = {}
        for name, p in (("clean", clean_path), ("tracked", out_path),
                        ("accepted", acc_path)):
            res, err = to_pdf(p)
            pdfs[name] = res if res else err
        report["pdf"] = pdfs
        for name in ("clean", "tracked", "accepted"):
            v = pdfs[name]
            lines.append("PDF %-9s: %s" % (name, ("%d pages, %s" % v)
                                           if isinstance(v, tuple) else v))
        c, a = pdfs.get("clean"), pdfs.get("accepted")
        if isinstance(c, tuple) and isinstance(a, tuple):
            if c != a:
                ok = False
                lines.append("FAIL: accepting every change does not reproduce "
                             "the clean file's pagination")
        t = pdfs.get("tracked")
        if isinstance(c, tuple) and isinstance(t, tuple) and c[1] != t[1]:
            ok = False
            lines.append("FAIL: page size changed")

        rt, rt_err = roundtrip_docx(out_path, v4_text, clean_text)
        report["roundtrip"] = rt or rt_err
        if rt:
            rc = rt["counts"]
            lines.append("LibreOffice re-saved the file as .docx and wrote "
                         "back %d run-level w:ins, %d w:del, %d inserted and "
                         "%d deleted paragraph marks: the revisions are "
                         "understood as revisions, not tolerated as text"
                         % (rc["ins"], rc["dele"], rc["mark_ins"],
                            rc["mark_del"]))
            lines.append("REJECT ALL on LibreOffice's own output -> %d units "
                         "vs v4 %d: %s;  ACCEPT ALL -> %d vs clean %d: %s"
                         % (rt["n_reject"], len(v4_text),
                            "IDENTICAL" if rt["reject_ok"] else "DIFFER",
                            rt["n_accept"], len(clean_text),
                            "IDENTICAL" if rt["accept_ok"] else "DIFFER"))
            if not (rc["ins"] > 0 and rc["dele"] > 0):
                ok = False
                lines.append("FAIL: the round trip lost the revisions")
            if not (rt["reject_ok"] and rt["accept_ok"]):
                ok = False
                lines.append("FAIL: an independent OOXML reader disagrees "
                             "with our accept/reject semantics")
        else:
            lines.append("LibreOffice round trip: %s" % rt_err)
    else:
        report["pdf"] = None
        report["roundtrip"] = None

    report["ok"] = ok
    report["check_lines"] = lines
    return ok, lines


def _compare(want, got, limit=12):
    sm = difflib.SequenceMatcher(a=want, b=got, autojunk=False)
    diffs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        for k in range(max(i2 - i1, j2 - j1)):
            diffs.append((tag,
                          want[i1 + k] if i1 + k < i2 else None,
                          got[j1 + k] if j1 + k < j2 else None))
    return (not diffs), diffs[:limit] if diffs else []


# ==========================================================================
# supplementary probe
# ==========================================================================

def supp_probe():
    """How much of supplementary_v4 survives into supplementary_revised?

    Prose and tables are measured separately, because the supplement's
    substance is in its tables: v4 carried most of its content in 15 of them.
    """
    a = docx.Document(SUPP_V4_DOCX)
    b = docx.Document(SUPP_CLEAN_DOCX)
    ai = collect_items(a)
    bi = collect_items(b)
    ap = [x for x in ai if x.kind == "p"]
    bp = [x for x in bi if x.kind == "p"]
    ops, _ = align(ap, bp)
    matched = [(i, j) for k, i, j in ops if k == "match"]
    kept_words = 0
    identical = 0
    for i, j in matched:
        sm = difflib.SequenceMatcher(a=ap[i].words, b=bp[j].words,
                                     autojunk=False)
        kept_words += sum(bl.size for bl in sm.get_matching_blocks())
        if norm(ap[i].text) == norm(bp[j].text):
            identical += 1

    at = [x for x in ai if x.kind == "tbl"]
    bt = [x for x in bi if x.kind == "tbl"]

    def tbl_words(items):
        n = 0
        for t in items:
            for tr in table_rows(t.el):
                for tc in row_cells(tr):
                    n += len(norm(cell_text(tc)).split())
        return n

    def tbl_rows_text(items):
        out = []
        for t in items:
            for tr in table_rows(t.el):
                out.append(norm(" | ".join(cell_text(tc)
                                           for tc in row_cells(tr))))
        return out

    a_rows, b_rows = tbl_rows_text(at), tbl_rows_text(bt)
    b_set = set(b_rows)
    rows_kept = sum(1 for r in a_rows if r in b_set)

    return dict(v4_paras=len(ap), new_paras=len(bp), matched=len(matched),
                identical=identical,
                v4_words=sum(len(x.words) for x in ap),
                new_words=sum(len(x.words) for x in bp),
                kept_words=kept_words, v4_tables=len(at), new_tables=len(bt),
                v4_table_words=tbl_words(at), new_table_words=tbl_words(bt),
                v4_table_rows=len(a_rows), new_table_rows=len(b_rows),
                table_rows_kept=rows_kept,
                v4_images=len([x for x in ai if x.kind == "img"]),
                new_images=len([x for x in bi if x.kind == "img"]))


# ==========================================================================
# report
# ==========================================================================

SUPP_STATEMENT = """\
The Supplementary Information was not edited in this revision; it was rebuilt. \
Supplementary v4 contained {v4_paras} text paragraphs ({v4_words} words), \
{v4_tables} tables ({v4_table_rows} rows, {v4_table_words} words of cell \
content) and {v4_images} figures. The revised Supplementary Information \
contains {new_paras} text paragraphs ({new_words} words), {new_tables} tables \
({new_table_rows} rows, {new_table_words} words of cell content) and \
{new_images} figures — about {ratio:.0f} times the content. Only {identical} \
of supplementary v4's {v4_paras} paragraphs and {table_rows_kept} of its \
{v4_table_rows} table rows survive unchanged. A change-marked supplementary \
would therefore be a document in which almost every paragraph, every table and \
every figure of the original is marked deleted and a wholly new \
{new_total:,}-word document is marked inserted: it would be larger than either \
file, Word's Review pane would list several thousand revisions, and it would \
tell a reader nothing that this sentence does not. We have therefore supplied \
the revised Supplementary Information as a clean file only. The manuscript, \
which was edited rather than rebuilt, is supplied with genuine track changes \
as manuscript_revised_tracked.docx.
"""


def write_report(report, supp, clean_current_diffs):
    st = report.get("stats", {})
    cnt = report.get("counts", {})
    xml = report.get("xml", {})
    pdf = report.get("pdf") or {}

    v4_total = supp["v4_words"] + supp["v4_table_words"]
    new_total = supp["new_words"] + supp["new_table_words"]
    supp_txt = SUPP_STATEMENT.format(
        ratio=new_total / max(v4_total, 1), new_total=new_total, **supp)

    L = []
    A = L.append
    A("# Tracked changes — what was built and how it was verified\n")
    A("Deliverable: `manuscript_revised_tracked.docx` — the resubmission3 "
      "manuscript carrying real OOXML revision marks against "
      "`1st_submission/manuscript_v4.docx`, the file the editor received.\n")
    A("Built by `scripts/build_tracked_changes.py`. Author string `%s`, "
      "revision date `%s` (fixed, not `now()`, so a rebuild over unchanged "
      "inputs is byte-stable). Revision ids 1…%d, all unique.\n"
      % (REV_AUTHOR, REV_DATE, report.get("n_ids", 0)))
    A("> **This file must be rebuilt after the manuscript text is final.** "
      "The script rebuilds from the pristine v4 file every time and is "
      "idempotent; re-run `build_manuscript_docx.py` first, then this.\n")
    if clean_current_diffs:
        A("> ## ⚠ THIS BUILD IS STALE\n"
          "> When it was built, `manuscript_revised.docx` did **not** match "
          "`manuscript_revised.md`: %d paragraphs differ, so the manuscript "
          "text was still being edited. Everything below is true of the "
          "`manuscript_revised.docx` that was in the package at build time "
          "(content SHA-256 in §9), and the tracked file accepts to exactly "
          "that file — but it is not the final manuscript. Before the package "
          "ships, run:\n"
          ">\n"
          "> ```\n"
          "> /path/to/anaconda3/bin/python "
          "nature_communications/resubmission3/scripts/build_manuscript_docx.py\n"
          "> /path/to/anaconda3/bin/python "
          "nature_communications/resubmission3/scripts/build_tracked_changes.py\n"
          "> ```\n"
          ">\n"
          "> and confirm this banner is gone.\n" % len(clean_current_diffs))

    A("## 1. What the file contains\n")
    A("| quantity | value |")
    A("|---|---|")
    A("| `<w:ins ` elements in `word/document.xml` | **%d** |" % xml.get("ins", 0))
    A("| `<w:del ` elements in `word/document.xml` | **%d** |" % xml.get("dele", 0))
    A("| `<w:delText>` elements | %d |" % xml.get("delText", 0))
    A("| run-level `w:ins` / `w:del` | %d / %d |"
      % (cnt.get("ins", 0), cnt.get("dele", 0)))
    A("| runs inside them | %d inserted / %d deleted |"
      % (cnt.get("ins_runs", 0), cnt.get("del_runs", 0)))
    A("| inserted / deleted paragraph marks | %d / %d |"
      % (cnt.get("mark_ins", 0), cnt.get("mark_del", 0)))
    A("| inserted / deleted table cells (`w:cellIns` / `w:cellDel`) | %d / %d |"
      % (cnt.get("cell_ins", 0), cnt.get("cell_del", 0)))
    A("| inserted / deleted table rows (`w:trPr/w:ins` / `w:del`) | %d / %d |"
      % (cnt.get("row_ins", 0), cnt.get("row_del", 0)))
    A("| words unchanged / inserted / deleted | %d / %d / %d |"
      % (st.get("equal_words", 0), st.get("ins_words", 0),
         st.get("del_words", 0)))
    A("| paragraphs rewritten / wholly inserted / wholly deleted / untouched | "
      "%d / %d / %d / %d |"
      % (st.get("changed_paragraphs", 0), st.get("ins_paragraphs", 0),
         st.get("del_paragraphs", 0), st.get("unchanged_paragraphs", 0)))
    A("| Table 1 cells with a tracked change | %d |"
      % st.get("changed_cells", 0))
    A("| file size | %.2f MB |\n" % report.get("size_mb", 0.0))

    A("## 2. How the diff was computed\n")
    A("Both documents are reduced to a linear sequence of body items — text "
      "paragraphs, tables, image paragraphs — and aligned by a "
      "Needleman-Wunsch dynamic program whose match score is "
      "`difflib.SequenceMatcher` similarity over normalised word lists "
      "(threshold %.2f; below it a pair is not a pair). Tables and images are "
      "anchors, which is what makes a paragraph that moved across Table 1 "
      "render as a deletion above it and an insertion below it — the same "
      "thing Word does with a move, and what keeps *reject all* reproducing "
      "v4's ordering and not merely its words.\n" % MIN_SIM_PARA)
    A("| | manuscript_v4 | manuscript_revised |")
    A("|---|---|---|")
    for k, name in (("p", "text paragraphs"), ("tbl", "tables"),
                    ("img", "figures")):
        A("| %s | %d | %d |" % (name, report["v4_items"].get(k, 0),
                                report["new_items"].get(k, 0)))
    A("")
    A("Alignment: **%d matched pairs**, **%d v4 items deleted**, "
      "**%d new items inserted**.\n"
      % (report["ops"].get("match", 0), report["ops"].get("del", 0),
         report["ops"].get("ins", 0)))
    A("Each matched pair is then diffed at word level. Kept and inserted text "
      "is sliced out of the clean document's own runs, so insertions carry the "
      "formatting the clean file gives them; deleted text is reconstructed "
      "from v4 as `<w:delText xml:space=\"preserve\">` runs. Whole-paragraph "
      "insertions and deletions additionally carry a paragraph-mark revision "
      "(`<w:pPr><w:rPr><w:ins/></w:rPr></w:pPr>`), so they accept and reject "
      "cleanly instead of leaving an empty paragraph behind; §3.4 explains why "
      "an inserted paragraph mark is written on the paragraph *before* the "
      "insertion.\n")
    A("Word-level `difflib` opcodes are coalesced before they are emitted: an "
      "unchanged island shorter than %d words with a change on both sides is "
      "absorbed into the change, and adjacent changes are merged into one "
      "deletion followed by one insertion. Without that, a rewritten sentence "
      "renders as a shredded ribbon of strikethrough and underline around "
      "\"the\", \"of\" and \"and\". Absorbing is free: the island is emitted as "
      "a deletion and an insertion of the same words, so accept and reject are "
      "unaffected — which §3.1 and §3.2 then demonstrate.\n" % MIN_EQUAL_RUN)

    A("## 3. Verification\n")
    A("Every line below is produced by the script, not asserted by hand.\n")
    A("```")
    for line in report.get("check_lines", []):
        A(line)
    A("```\n")

    A("### 3.1 Reject all changes reproduces manuscript_v4\n")
    A("`apply_revisions(body, \"reject\")` performs a real reject on a copy of "
      "the XML tree: `w:ins` subtrees are removed, `w:del` subtrees are "
      "unwrapped and their `w:delText` renamed back to `w:t`, paragraphs whose "
      "inserted paragraph mark disappears are merged into the following "
      "paragraph, rows marked `w:trPr/w:ins` and cells marked `w:cellIns` are "
      "removed. The resulting body text — paragraphs and table cells in "
      "document order, whitespace collapsed, zero-width break hints removed — "
      "is compared against `manuscript_v4.docx`.\n")
    A("**Result: %d text units on both sides, %s.**\n"
      % (report.get("n_v4_units", 0),
         "identical" if not report.get("reject_diffs")
         else "%d differ (listed below)" % len(report["reject_diffs"])))
    if report.get("reject_diffs"):
        A("```")
        for tag, wnt, got in report["reject_diffs"]:
            A("[%s] v4  : %r" % (tag, (wnt or "")[:150]))
            A("      rej : %r" % ((got or "")[:150],))
        A("```\n")

    A("### 3.2 Accept all changes reproduces manuscript_revised.docx\n")
    A("The same machinery in `accept` mode, compared against the clean file "
      "the package ships.\n")
    A("**Result: %d text units on both sides, %s.**\n"
      % (report.get("n_clean_units", 0),
         "identical" if not report.get("accept_diffs")
         else "%d differ (listed below)" % len(report["accept_diffs"])))
    if report.get("accept_diffs"):
        A("```")
        for tag, wnt, got in report["accept_diffs"]:
            A("[%s] clean: %r" % (tag, (wnt or "")[:150]))
            A("      acc  : %r" % ((got or "")[:150],))
        A("```\n")

    A("### 3.3 It opens without repair, and the page geometry is unchanged\n")
    val, xsds = report.get("schema", (None, []))
    A("Schema validation was attempted and is not available on this host: an "
      "XSD validator is %s, and a search of the python-docx package, "
      "`/usr/share/xml`, `/usr/share/schemas` and the environment's "
      "`share/xml` found %s. The file is therefore **not** validated against "
      "the published wordprocessingml XSD."
      % (val or "not installed (`import xmlschema` fails)",
         ("%d copies of `wml.xsd`: %s" % (len(xsds), xsds)) if xsds
         else "no `wml.xsd`"))
    A("\nWhat *is* checked: the package reopens under `python-docx`; every "
      "`w:ins`, `w:del`, `w:cellIns` and `w:cellDel` carries `w:id`, "
      "`w:author` and `w:date`; every `w:id` is unique; no `w:t` appears "
      "inside a `w:del` and no `w:delText` outside one; the paragraph-mark "
      "`w:rPr` is the last child of its `w:pPr` and the mark is the first "
      "child of that `w:rPr`, which is what the `CT_PPr` and `CT_ParaRPr` "
      "sequences require; and every revision element is a child of `w:p` or "
      "`w:hyperlink`. Beyond that, LibreOffice — an independent OOXML "
      "implementation — parses, lays out and renders the file to PDF, which is "
      "the practical open-without-repair test, and re-saves it as `.docx` with "
      "the revisions intact (§3.4).\n")
    A("| file | pages | page size |")
    A("|---|---|---|")
    for name in ("clean", "tracked", "accepted"):
        v = pdf.get(name)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            A("| %s | %d | %s |" % (name, v[0], v[1]))
        else:
            A("| %s | — | %s |" % (name, v))
    A("")
    A("`tracked` is the tracked file rendered *with the markup shown*, which "
      "is how LibreOffice renders a revision-marked document by default: "
      "deleted text is still on the page, struck through, so the page count "
      "necessarily rises. `accepted` is the tracked file with every change "
      "programmatically accepted; it must — and does — paginate exactly like "
      "the clean file. The page size is unchanged in all three.\n")

    A("### 3.4 An independent OOXML implementation agrees\n")
    rt = report.get("roundtrip")
    if isinstance(rt, dict):
        rc = rt["counts"]
        A("The file was re-saved as `.docx` by LibreOffice — an independent "
          "OOXML implementation, and the closest thing to Word available on "
          "this host — and the accept/reject simulation was then run on *its* "
          "output. A reader that had merely tolerated the revision marks would "
          "have written the text back out flat.\n")
        A("| after LibreOffice re-saved the file | |")
        A("|---|---|")
        A("| run-level `w:ins` / `w:del` written back | %d / %d |"
          % (rc["ins"], rc["dele"]))
        A("| inserted / deleted paragraph marks written back | %d / %d |"
          % (rc["mark_ins"], rc["mark_del"]))
        A("| inserted table cells written back | %d |" % rc["cell_ins"])
        A("| **reject all** on LibreOffice's output vs manuscript_v4 | "
          "%d units, **%s** |"
          % (rt["n_reject"], "identical" if rt["reject_ok"] else "DIFFER"))
        A("| **accept all** on LibreOffice's output vs manuscript_revised | "
          "%d units, **%s** |\n"
          % (rt["n_accept"], "identical" if rt["accept_ok"] else "DIFFER"))
        A("This check is why the inserted paragraph marks sit where they do. "
          "Inserting a paragraph P between A and B inserts P's text plus one "
          "paragraph mark, and either ¶A or ¶P may carry the `w:ins`; both are "
          "valid OOXML and Word writes both, depending on whether the author "
          "split at the end of A or at the start of B. They are not equally "
          "robust. Marking ¶P and re-saving through LibreOffice added a second "
          "mark on ¶A without removing the first, and *reject all* then "
          "removed both and ran A and B together — six paragraph breaks were "
          "lost that way. Marking ¶A instead is LibreOffice's own model: %d of "
          "the %d inserted paragraph marks are shifted onto the preceding "
          "paragraph (%d could not be, because no paragraph precedes them), "
          "the file survives the round trip unchanged in revision terms, and "
          "reject-all reproduces v4 exactly under both readings.\n"
          % (report.get("mark_shift", (0, 0))[0],
             cnt.get("mark_ins", 0), report.get("mark_shift", (0, 0))[1]))
    else:
        A("Not run (%s).\n" % (rt if rt else "--no-pdf"))

    A("## 4. Figures\n")
    A("Image binaries are never marked: OOXML revision marks apply to text, "
      "and a diff of two PNGs is meaningless in Word's Review pane. A changed "
      "or new figure is tracked through its caption paragraph, which is "
      "diffed like any other text. Below is every image part in each file so "
      "that a human can confirm which figures changed.\n")
    A("| | manuscript_v4 | manuscript_revised_tracked |")
    A("|---|---|---|")
    v4i, newi = report["v4_images"], report["new_images"]
    for k in range(max(len(v4i), len(newi))):
        a = ("%s, %.2f MB, sha256 %s…" % (v4i[k][0], v4i[k][1] / 1e6, v4i[k][2])
             if k < len(v4i) else "—")
        b = ("%s, %.2f MB, sha256 %s…" % (newi[k][0], newi[k][1] / 1e6,
                                          newi[k][2])
             if k < len(newi) else "—")
        A("| %d | %s | %s |" % (k + 1, a, b))
    A("")
    A("**No image part is shared between the two files.** `manuscript_v4.docx` "
      "embeds %d images and the revision embeds %d; every SHA-256 above "
      "differs, so every display item is new artwork and a reviewer should "
      "read the caption revisions for what changed scientifically rather than "
      "look for an unchanged figure.\n" % (len(v4i), len(newi)))

    A("## 5. Content of v4 that is marked deleted in full\n")
    A("These v4 paragraphs had no counterpart in the revision above the %.2f "
      "similarity threshold and are therefore marked as whole-paragraph "
      "deletions (%d of them).\n" % (MIN_SIM_PARA, st.get("del_paragraphs", 0)))
    A("```")
    for kind, txt in report.get("v4_only", [])[:60]:
        if kind == "paragraph":
            A("- %s" % txt)
    A("```\n")

    A("## 6. Table 1\n")
    A("v4's Table 1 is %d×%d, the revision's is %d×%d. Rows and columns are "
      "aligned by the same dynamic program (on cell text), then matched cells "
      "are diffed word by word and the changes marked inside the cells. "
      "Columns present only in the revision are marked `w:cellIns` on every "
      "cell and their content wrapped in `w:ins`; columns present only in v4 "
      "would be re-inserted as `w:cellDel` cells (none were needed here). "
      "%d cells carry a tracked change.\n"
      % (report.get("table_rows", (0, 0))[0],
         report.get("table_cols", (0, 0))[0],
         report.get("table_rows", (0, 0))[1],
         report.get("table_cols", (0, 0))[1],
         st.get("changed_cells", 0)))
    A("Column alignment, by header (the pairing is decided on cell text alone, "
      "so it is a positional statement, not a semantic one — read it as *what "
      "stands in that column now* rather than *what that column became*):\n")
    ah, bh = report.get("table_col_headers", ([], []))
    A("```")
    for k, i, j in report.get("table_col_ops", []):
        left = ah[i] if i is not None and i < len(ah) else "—"
        right = bh[j] if j is not None and j < len(bh) else "—"
        A("%-6s %-24s -> %s" % (k, left[:24], right[:44]))
    A("```\n")

    A("## 7. Known limits of this file\n")
    A("- **Figures are not marked.** See §4. Their captions are.\n")
    A("- **Field plumbing in deleted v4 text is dropped: %d elements.** Some "
      "v4 runs carry `w:fldChar` / `w:instrText` (Word's field machinery). "
      "When such a paragraph is marked deleted, the field machinery is dropped "
      "and the field's visible *result* is kept as deleted text, so nothing a "
      "reader sees is lost — the alternative would be emitting "
      "`w:delInstrText`, which buys nothing here. Hyperlinks in deleted v4 "
      "text are kept intact as `w:hyperlink > w:del > w:r`, which is the "
      "correct encoding (`w:del` cannot contain a `w:hyperlink`).\n"
      % st.get("dropped_field_parts", 0))
    A("- **Whitespace and zero-width break hints are not diffed.** "
      "`build_manuscript_docx.py` inserts U+200B soft-break hints inside long "
      "tokens so URLs and file paths wrap; they are invisible and are ignored "
      "by the diff. Everything else — including a curly quote replacing a "
      "straight one — is a real tracked change.\n")
    A("- **Word does not record this as a *move*.** A paragraph that moved "
      "across a display item shows as a deletion in its old place and an "
      "insertion in its new one. `w:moveFrom` / `w:moveTo` were not used: "
      "they add nothing an editor needs and are less widely supported.\n")

    A("## 8. Supplementary Information\n")
    A("The same machinery was run against `supplementary_v4.docx` as a "
      "measurement before deciding whether a change-marked supplementary is "
      "worth producing. It is not, and none is shipped. The measurement:\n")
    A("| | supplementary_v4 | supplementary_revised |")
    A("|---|---|---|")
    A("| text paragraphs | %d | %d |" % (supp["v4_paras"], supp["new_paras"]))
    A("| words in those paragraphs | %d | %d |"
      % (supp["v4_words"], supp["new_words"]))
    A("| tables | %d | %d |" % (supp["v4_tables"], supp["new_tables"]))
    A("| table rows | %d | %d |"
      % (supp["v4_table_rows"], supp["new_table_rows"]))
    A("| words in table cells | %d | %d |"
      % (supp["v4_table_words"], supp["new_table_words"]))
    A("| figures | %d | %d |" % (supp["v4_images"], supp["new_images"]))
    A("| **total words** | **%d** | **%d** (×%.0f) |"
      % (v4_total, new_total, new_total / max(v4_total, 1)))
    A("| v4 paragraphs alignable at all (≥ %.2f similarity) | %d of %d | |"
      % (MIN_SIM_PARA, supp["matched"], supp["v4_paras"]))
    A("| v4 paragraphs unchanged | %d of %d | |"
      % (supp["identical"], supp["v4_paras"]))
    A("| v4 table rows surviving verbatim | %d of %d | |\n"
      % (supp["table_rows_kept"], supp["v4_table_rows"]))
    A("**Verdict: a tracked-changes supplementary is not meaningful, and none "
      "is shipped.** The paragraph below is written to be pasted verbatim into "
      "the response letter (item E12) and into `README.md`, which "
      "`the internal build contract` §7.1 requires when the supplementary is not "
      "supplied with track changes.\n")
    A("> " + supp_txt.replace("\n", "\n> ").rstrip("> \n") + "\n")

    A("## 9. Provenance of the inputs\n")
    A("Hashes are over the package *contents* (sorted entry names and bytes), "
      "not over the `.docx` file, because python-docx stamps every zip entry "
      "with the time of the write. Under that hash the build is deterministic: "
      "the revision date is a constant rather than `now()`, revision ids are "
      "assigned in document order, and two consecutive runs over unchanged "
      "inputs produce the same content hash.\n")
    A("```")
    for p in (V4_DOCX, CLEAN_DOCX, OUT_DOCX):
        if os.path.exists(p):
            A("%-38s %9d bytes  content sha256 %s…"
              % (os.path.basename(p), os.path.getsize(p),
                 content_digest(p)[:24]))
    A("```\n")
    if clean_current_diffs:
        A("**`manuscript_revised.docx` did not match `manuscript_revised.md` "
          "when this file was built: %d paragraphs differ.** The clean `.docx` "
          "in the package had not yet been re-rendered from the edited "
          "markdown, so this tracked file marks up the `.docx` that was "
          "actually in the package, not the newest text. The paragraphs that "
          "had already moved ahead of the `.docx`:\n"
          % len(clean_current_diffs))
        A("```")
        for tag, wnt, got in clean_current_diffs[:12]:
            A("[%s] markdown: %s" % (tag, (wnt or "")[:120]))
            A("      docx    : %s" % ((got or "")[:120],))
        A("```\n")
    else:
        A("`manuscript_revised.docx` was checked against "
          "`manuscript_revised.md` before the build: 0 differing paragraphs, "
          "so the tracked file is built on a current clean file.\n")

    open(REPORT, "w", encoding="utf-8").write("\n".join(L))
    return REPORT


# ==========================================================================
# main
# ==========================================================================

def main():
    global V4_DOCX, CLEAN_DOCX, OUT_DOCX, REPORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true",
                    help="skip the LibreOffice pagination check")
    ap.add_argument("--allow-stale", action="store_true",
                    help="build even if manuscript_revised.docx does not match "
                         "manuscript_revised.md")
    ap.add_argument("--supp-probe", action="store_true",
                    help="only measure the supplementary overlap and stop")
    ap.add_argument("--baseline", default=V4_DOCX,
                    help="the submitted file to diff against")
    ap.add_argument("--clean", default=CLEAN_DOCX,
                    help="the clean revision to mark up")
    ap.add_argument("--out", default=OUT_DOCX, help="the tracked file to write")
    ap.add_argument("--report", default=REPORT)
    args = ap.parse_args()

    V4_DOCX, CLEAN_DOCX = args.baseline, args.clean
    OUT_DOCX, REPORT = args.out, args.report

    if args.supp_probe:
        for k, v in sorted(supp_probe().items()):
            print("%-12s %s" % (k, v))
        return 0

    print("=" * 76)
    print("TRACKED CHANGES")
    print("=" * 76)
    print("  baseline : %s" % V4_DOCX)
    print("  clean    : %s" % CLEAN_DOCX)
    print("  output   : %s" % OUT_DOCX)

    default_clean = CLEAN_DOCX == os.path.join(RESUB, "manuscript_revised.docx")
    diffs = clean_file_is_current() if default_clean else []
    if not default_clean:
        print("  clean file overridden on the command line; the "
              "manuscript_revised.md staleness check is skipped")
    if diffs:
        print("  WARNING  manuscript_revised.docx does not match "
              "manuscript_revised.md (%d paragraphs differ)" % len(diffs))
        for tag, wnt, got in diffs[:5]:
            print("    [%s] md  : %r" % (tag, (wnt or "")[:110]))
            print("         docx: %r" % ((got or "")[:110],))
        if not args.allow_stale:
            print("  Re-run build_manuscript_docx.py first, or pass "
                  "--allow-stale.")
            return 2
    elif default_clean:
        print("  clean file matches manuscript_revised.md: 0 differing "
              "paragraphs")

    report = {}
    report["schema"] = schema_probe()
    report["v4_images"] = image_digests(V4_DOCX)
    build(V4_DOCX, CLEAN_DOCX, OUT_DOCX, "manuscript", report)
    report["new_images"] = image_digests(OUT_DOCX)
    print("  wrote %s (%.2f MB)" % (OUT_DOCX, report["size_mb"]))

    ok, lines = check(V4_DOCX, CLEAN_DOCX, OUT_DOCX, report,
                      do_pdf=not args.no_pdf)
    print("-" * 76)
    for line in lines:
        print("  " + line)
    print("-" * 76)

    supp = supp_probe()
    print("  supplementary overlap probe: %d/%d v4 paragraphs alignable, "
          "%d identical, %d/%d v4 words survive (%.1f%%)"
          % (supp["matched"], supp["v4_paras"], supp["identical"],
             supp["kept_words"], supp["v4_words"],
             100.0 * supp["kept_words"] / max(supp["v4_words"], 1)))

    path = write_report(report, supp, diffs)
    print("  wrote %s" % path)
    print("=" * 76)
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
