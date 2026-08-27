#!/usr/bin/env python
"""Build cover_letter.docx from cover_letter.md.

Plain business-letter layout, no template. A4 page (matching the rest of the
resubmission package, which inherits A4 from the submitted v4 Word files),
1 inch margins, Times New Roman 11 pt.

The Markdown source is deliberately simple, so this parser handles only what it
uses: `**bold**` runs, `*italic*` runs, numbered list items, and blank-line
paragraph separation. Anything else is emitted as plain text.

Run with the only interpreter that carries python-docx 1.2.0:

    /path/to/conda/bin/python \
        nature_communications/resubmission/scripts/build_cover_letter_docx.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "cover_letter.md"
DST = ROOT / "cover_letter.docx"

FONT = "Times New Roman"
SIZE = Pt(11)

# A4, as used by manuscript_revised.docx, supplementary_revised.docx and
# response_to_reviewers.docx.  python-docx defaults to US Letter.
A4_WIDTH = Inches(8.268)
A4_HEIGHT = Inches(11.693)
MARGIN = Inches(1.0)

# Paragraph spacing is deliberately tight so the letter holds to a single A4
# page at 11 pt without reducing the font size.  The added compliance
# paragraph pushes the signature block onto a second page; the letter is left
# at two pages rather than tightened further, which is normal for a
# resubmission cover letter.
SPACE_AFTER = Pt(3)
LINE_SPACING = Pt(11.8)

# Lines that make up the address / signature blocks: no extra spacing after,
# because they read as a block rather than as separate paragraphs.
TIGHT_LINES = {
    "The Editor",
    "*Nature Communications*",
    "Corresponding author",
    "Institute for Food Safety and Health, Illinois Institute of Technology",
    "Bedford Park, IL 60501, United States · rtian@illinoistech.edu",
    "**Renmao Tian**, corresponding author",
}

INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*)", re.DOTALL)


def set_base_style(doc: Document) -> None:
    style = doc.styles["Normal"]
    style.font.name = FONT
    style.font.size = SIZE
    # East-Asian font mapping, so Word does not silently substitute.
    style.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    fmt = style.paragraph_format
    fmt.space_before = Pt(0)
    fmt.space_after = SPACE_AFTER
    fmt.line_spacing = LINE_SPACING


def set_a4(doc: Document) -> None:
    for section in doc.sections:
        section.page_width = A4_WIDTH
        section.page_height = A4_HEIGHT
        section.left_margin = MARGIN
        section.right_margin = MARGIN
        section.top_margin = MARGIN
        section.bottom_margin = MARGIN


def add_runs(paragraph, text: str) -> None:
    """Emit `text` into `paragraph`, honouring **bold** and *italic* spans."""
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            run = paragraph.add_run(piece[2:-2])
            run.bold = True
        elif piece.startswith("*") and piece.endswith("*"):
            run = paragraph.add_run(piece[1:-1])
            run.italic = True
        else:
            run = paragraph.add_run(piece)
        run.font.name = FONT
        run.font.size = SIZE
        run.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)


def build() -> Document:
    doc = Document()
    set_a4(doc)
    set_base_style(doc)

    blocks = [b.strip("\n") for b in SRC.read_text(encoding="utf-8").split("\n\n")]

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue

            list_item = re.match(r"^(\d+)\.\s+(.*)$", line)
            para = doc.add_paragraph()
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT

            if list_item:
                para.paragraph_format.left_indent = Inches(0.35)
                para.paragraph_format.space_after = Pt(2)
                add_runs(para, f"{list_item.group(1)}. {list_item.group(2)}")
            else:
                if line in TIGHT_LINES:
                    para.paragraph_format.space_after = Pt(0)
                add_runs(para, line)

    return doc


def main() -> None:
    doc = build()
    doc.save(DST)

    # Re-open to confirm the file parses, and report what is in it.
    check = Document(DST)
    section = check.sections[0]
    print(f"wrote {DST}")
    print(f"paragraphs: {len(check.paragraphs)}")
    print(
        "page size: "
        f"{section.page_width.inches:.3f} x {section.page_height.inches:.3f} in "
        f"({section.page_width.emu / 12700:.1f} x {section.page_height.emu / 12700:.1f} pt)"
    )
    words = sum(len(p.text.split()) for p in check.paragraphs)
    print(f"words (all paragraphs): {words}")


if __name__ == "__main__":
    main()
