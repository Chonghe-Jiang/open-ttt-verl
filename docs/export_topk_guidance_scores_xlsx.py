#!/usr/bin/env python3
"""Export per-step and cumulative-best training scores to a dependency-free XLSX."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


SCORE_RE = re.compile(r"step:(\d+).*?critic/score/max:([0-9.eE+-]+)")


def scores_from_log(log_path: Path) -> list[tuple[int, float, float]]:
    best = float("-inf")
    rows: list[tuple[int, float, float]] = []
    for line in log_path.read_text(errors="replace").splitlines():
        match = SCORE_RE.search(line)
        if match:
            step, score = int(match.group(1)), float(match.group(2))
            best = max(best, score)
            rows.append((step, score, best))
    if not rows:
        raise ValueError("No completed training-step scores found in log")
    return rows


def cell(ref: str, value: str, *, style: int = 0, text: bool = False) -> str:
    if text:
        return f'<c r="{ref}" t="inlineStr" s="{style}"><is><t>{escape(value)}</t></is></c>'
    return f'<c r="{ref}" s="{style}"><v>{value}</v></c>'


def write_xlsx(rows: list[tuple[int, float, float]], output_path: Path) -> None:
    sheet_rows = [
        '<row r="1">'
        + cell("A1", "Step", style=1, text=True)
        + cell("B1", "Step Maximum Score", style=1, text=True)
        + cell("C1", "Historical Maximum Score", style=1, text=True)
        + "</row>"
    ]
    for row_num, (step, step_max, historical_max) in enumerate(rows, start=2):
        sheet_rows.append(
            f'<row r="{row_num}">'
            + cell(f"A{row_num}", str(step))
            + cell(f"B{row_num}", f"{step_max:.8f}", style=2)
            + cell(f"C{row_num}", f"{historical_max:.8f}", style=2)
            + "</row>"
        )
    sheet = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" state="frozen"/></sheetView></sheetViews>
  <cols><col min="1" max="1" width="12" customWidth="1"/><col min="2" max="3" width="27" customWidth="1"/></cols>
  <sheetData>{''.join(sheet_rows)}</sheetData>
  <autoFilter ref="A1:C{len(rows) + 1}"/>
</worksheet>'''
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="1"><numFmt numFmtId="164" formatCode="0.0000"/></numFmts>
  <fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="1" borderId="0" xfId="0"/><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs>
</styleSheet>'''
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>'''
    workbook = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Score History" sheetId="1" r:id="rId1"/></sheets>
</workbook>'''
    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''
    workbook_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", styles)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_xlsx(scores_from_log(args.log), args.output)


if __name__ == "__main__":
    main()
