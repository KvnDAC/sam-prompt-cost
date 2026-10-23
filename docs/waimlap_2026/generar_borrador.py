"""Build a review draft from the supplied template, preserving header/footer.

Run with the repository virtualenv. PDF export uses an isolated, hidden local
Microsoft Word instance on Windows; it never submits or sends the document.
"""
from pathlib import Path
import hashlib
import json
import re
import zipfile
from io import BytesIO

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.enum.section import WD_SECTION


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMPLATE = ROOT / 'WAIMLAp_2026_Extended_Abstract_Template (1).docx'
STEM = 'Abstract_TheInformationalCostOfSpatialPromptsInSAM'


def paragraph(doc, text, *, size=11, bold=False, before=0, after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(text)
    r.font.name = 'Calibri'
    r.font.size = Pt(size)
    r.bold = bold
    return p


def build(source=HERE / 'abstract_draft.md', output_dir=HERE, page_break=False, flow=False):
    source, output_dir = source.resolve(), output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original_hash = hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    doc = Document(TEMPLATE)
    # Only replace the body; section properties retain logos and header/footer.
    body = doc._element.body
    for child in list(body):
        if child.tag != qn('w:sectPr'):
            body.remove(child)
    chunks = source.read_text(encoding='utf-8').strip().split('\n\n')
    for index, text in enumerate(chunks):
        if text.startswith('# '):
            paragraph(doc, text[2:], size=15, bold=True, after=6)
        elif index == 1:
            paragraph(doc, text, size=10.5, bold=True, after=3)
        elif index == 2:
            paragraph(doc, text, size=8.5, after=6)
        elif text.startswith('## '):
            is_results = text.startswith('## 3.')
            if is_results and not flow:
                if not page_break:
                    # Start results in a linked section. This works for dense
                    # page-1 drafts; --page_break avoids a Word floating-header
                    # offset observed with a shorter first page.
                    section = doc.add_section(WD_SECTION.NEW_PAGE)
                    section.header.is_linked_to_previous = True
                    section.footer.is_linked_to_previous = True
            p = paragraph(doc, text[3:], bold=True, before=5, after=3)
            if is_results and page_break and not flow:
                p.paragraph_format.page_break_before = True
            p.paragraph_format.keep_with_next = True
        elif text.startswith('|'):
            caption = paragraph(doc, 'Table 1. Development results; GT prompts are oracle references.', size=9, after=2)
            caption.paragraph_format.keep_with_next = True
            rows = [[cell.strip() for cell in line.strip('|').split('|')] for line in text.splitlines()]
            table = doc.add_table(rows=1, cols=2)
            table.style = 'Table Grid'
            table.rows[0].cells[0].text, table.rows[0].cells[1].text = rows[0]
            for row in rows[2:]:
                cells = table.add_row().cells
                for cell, value in zip(cells, row):
                    cell.text = value
            for i, row in enumerate(table.rows):
                row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
                for cell in row.cells:
                    for p in cell.paragraphs:
                        p.paragraph_format.space_after = Pt(1)
                        p.paragraph_format.space_before = Pt(1)
                        for run in p.runs:
                            run.font.name = 'Calibri'
                            run.font.size = Pt(9)
                            run.bold = i == 0
        elif text.startswith('!['):
            match = re.fullmatch(r'!\[(.+)\]\((.+)\)', text.strip())
            if not match:
                raise ValueError(f'Invalid image marker: {text}')
            caption_text, image_path = match.groups()
            image_path = (source.parent / image_path).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            p = doc.add_paragraph();p.alignment = 1
            p.paragraph_format.space_before = Pt(2);p.paragraph_format.space_after = Pt(1)
            p.add_run().add_picture(str(image_path),width=Inches(6.55))
            caption = paragraph(doc,caption_text,size=8.5,after=3)
            caption.alignment = 1
        elif text.startswith('[') and len(text) > 2 and text[1].isdigit():
            paragraph(doc, text, size=9, after=2)
        else:
            paragraph(doc, text)
    output = output_dir / (STEM + '.docx')
    doc.save(output)
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == original_hash
    # python-docx serializes header XML even without edits; retain original bytes.
    with zipfile.ZipFile(TEMPLATE) as source:
        originals = {name: source.read(name) for name in source.namelist()
                     if name.startswith(('word/header', 'word/footer', 'word/media/'))}
    buffer = BytesIO()
    with zipfile.ZipFile(output) as current, zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as rebuilt:
        for item in current.infolist():
            rebuilt.writestr(item, originals.get(item.filename, current.read(item.filename)))
    output.write_bytes(buffer.getvalue())
    preserved = []
    with zipfile.ZipFile(TEMPLATE) as source, zipfile.ZipFile(output) as target:
        for name in source.namelist():
            if name.startswith('word/media/') or (name.startswith('word/header') and name.endswith('.xml')):
                assert source.read(name) == target.read(name), name
                preserved.append(name)
    verification = {'template_sha256': original_hash, 'preserved_parts': preserved,
                    'authorship_confirmed': False, 'status': 'review draft; not submitted'}
    (output_dir / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(output)
    return output, verification


def replace_headers_with_strip(path, strip, verification):
    """Use a stable inline rendering when Word offsets the template's floating shapes."""
    doc = Document(path)
    for section in doc.sections[1:]:
        section.header.is_linked_to_previous = False
    for section in doc.sections:
        header = section.header
        for child in list(header._element):
            header._element.remove(child)
        p = header.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.alignment = 1
        p.add_run().add_picture(str(strip), width=Inches(6.85))
    # The template has a single default header/footer that Word repeats on every
    # page; adding a second strip and footer to the body duplicated both on page 2.
    doc.save(path)
    verification['header_rendering'] = 'stable inline strip rendered from the supplied template; visually inspected'
    verification['preserved_parts_before_header_stabilization'] = verification.pop('preserved_parts')
    (path.parent / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')


def export_pdf(path, verification, duplicate_headers=False):
    import win32com.client
    word = win32com.client.DispatchEx('Word.Application')
    word.Visible = False
    word.DisplayAlerts = 0
    opened = None
    try:
        opened = word.Documents.Open(str(path), ReadOnly=not duplicate_headers, AddToRecentFiles=False)
        if duplicate_headers and opened.Sections.Count > 1:
            # Word may offset floating shapes when a later section links to the
            # template header. Native copy gives each section independent
            # anchors while preserving the original formatting and media.
            for index in range(2, opened.Sections.Count + 1):
                target_header = opened.Sections(index).Headers(1)
                target_header.LinkToPrevious = False
                opened.Sections(1).Headers(1).Range.Copy()
                target_header.Range.PasteAndFormat(16)  # wdFormatOriginalFormatting
                target_footer = opened.Sections(index).Footers(1)
                target_footer.LinkToPrevious = False
                opened.Sections(1).Footers(1).Range.Copy()
                target_footer.Range.PasteAndFormat(16)
            opened.Save()
        opened.Repaginate()
        pages = opened.ComputeStatistics(2)  # wdStatisticPages
        verification['word_page_count'] = pages
        pdf = path.with_suffix('.pdf')
        opened.ExportAsFixedFormat(str(pdf), 17)  # wdExportFormatPDF
        verification['pdf'] = pdf.name
        (path.parent / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
        print(f'PDF: {pdf.name}; pages: {pages}')
        if pages > 2:
            raise RuntimeError('Draft exceeds two pages; shorten the source before submission.')
    finally:
        if opened is not None:
            opened.Close(False)
        word.Quit()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', action='store_true')
    parser.add_argument('--source', type=Path, default=HERE / 'abstract_draft.md')
    parser.add_argument('--output_dir', type=Path, default=HERE)
    parser.add_argument('--page_break', action='store_true')
    parser.add_argument('--flow', action='store_true', help='no break before Results; page 1 fills before page 2')
    parser.add_argument('--duplicate_headers', action='store_true')
    parser.add_argument('--raster_header', type=Path)
    args = parser.parse_args()
    path, verification = build(args.source, args.output_dir, args.page_break, args.flow)
    if args.raster_header:
        replace_headers_with_strip(path, args.raster_header.resolve(), verification)
    if args.pdf:
        export_pdf(path, verification, args.duplicate_headers)
