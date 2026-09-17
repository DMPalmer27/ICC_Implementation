"""
File: extract_pdf_text.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Extracts plain text from the PDFs in Written_Resources/ into
    Written_Resources/extracted/, so the sources can be grepped and quoted without
    reopening a viewer.

    The extracted text is checked in deliberately. Several claims in docs/ cite a specific
    line of a specific paper (the base-q logarithm in the ICC paper's Appendix A, the
    lambda(q,d,m) notation in the 2024 paper), and those citations are only verifiable if
    the text is in the repository next to them.

    Two caveats about the output, both visible in the existing files. Mathematical layout
    does not survive: a displayed fraction becomes two lines, subscripts run together, and
    ligatures land as odd characters -- so the text is for locating and quoting prose, not
    for reading equations, which still need the PDF. And the files are byte sequences that
    grep may treat as binary; use `grep -a`, or read them in Python with
    errors='replace', as this script writes them.

Usage:
    .venv/bin/python tools/extract_pdf_text.py                  # any PDF missing a .txt
    .venv/bin/python tools/extract_pdf_text.py --force          # re-extract everything
"""

import _path  # noqa: F401  -- puts icc/ on sys.path; must precede the local imports

import argparse
from pathlib import Path

import pypdf

SOURCE_DIR = Path(__file__).resolve().parent.parent / "Written_Resources"
OUTPUT_DIR = SOURCE_DIR / "extracted"


def extract(pdf_path: Path, out_path: Path) -> int:
    """
    Writes one PDF's text, with a page marker before each page.

    The page markers matter: they are what lets a citation in docs/ say "page 5 of the
    2024 paper" and be checkable against the text file.

    :param pdf_path: PDF to read
    :param out_path: Destination .txt path
    :return: Number of pages extracted
    """
    reader = pypdf.PdfReader(str(pdf_path))
    chunks = []
    for number, page in enumerate(reader.pages, start=1):
        chunks.append(f"\n===== PAGE {number} =====\n")
        chunks.append(page.extract_text() or "")

    out_path.write_text("".join(chunks), encoding="utf-8")
    return len(reader.pages)


def main():
    parser = argparse.ArgumentParser(description="Extract text from Written_Resources PDFs")
    parser.add_argument("--force", action="store_true",
                        help="re-extract even when the .txt already exists")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    for pdf_path in sorted(SOURCE_DIR.glob("*.pdf")):
        out_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"
        if out_path.exists() and not args.force:
            print(f"  skip   {pdf_path.name}  ({out_path.name} exists)")
            continue
        pages = extract(pdf_path, out_path)
        print(f"  wrote  {out_path.name}  ({pages} pages, "
              f"{out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
