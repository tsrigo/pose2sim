#!/usr/bin/env python3
"""
Export a basic Markdown document to PDF with ReportLab.

Supported elements:
- headings (#, ##, ###)
- paragraphs
- unordered and ordered lists
- fenced code blocks
- pipe tables
- local images: ![alt](path)

This keeps dependencies light and is sufficient for repo docs that mix
Chinese text, simple examples, and generated figures.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont


ROOT = Path(__file__).resolve().parents[1]
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 2.0 * cm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN


def register_fonts() -> dict[str, str]:
    sans_regular = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    sans_bold = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    mono_regular = "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"

    try:
        pdfmetrics.registerFont(TTFont("DocSans", sans_regular))
        pdfmetrics.registerFont(TTFont("DocSansBold", sans_bold))
        pdfmetrics.registerFont(TTFont("DocMono", mono_regular))
        return {"sans": "DocSans", "bold": "DocSansBold", "mono": "DocMono"}
    except Exception:
        # Fallback: built-in CJK font for body text; Courier for code.
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return {"sans": "STSong-Light", "bold": "STSong-Light", "mono": "Courier"}


def build_styles(fonts: dict[str, str]) -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName=fonts["sans"],
        fontSize=10.5,
        leading=16,
        spaceAfter=8,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName=fonts["bold"],
        fontSize=20,
        leading=26,
        spaceBefore=6,
        spaceAfter=14,
        textColor=colors.HexColor("#1f2937"),
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName=fonts["bold"],
        fontSize=15,
        leading=20,
        spaceBefore=8,
        spaceAfter=10,
        textColor=colors.HexColor("#111827"),
    )
    h3 = ParagraphStyle(
        "H3",
        parent=styles["Heading3"],
        fontName=fonts["bold"],
        fontSize=12.5,
        leading=18,
        spaceBefore=6,
        spaceAfter=8,
        textColor=colors.HexColor("#111827"),
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=16,
        firstLineIndent=-10,
        bulletIndent=0,
        spaceAfter=4,
    )
    number = ParagraphStyle(
        "Number",
        parent=body,
        leftIndent=18,
        firstLineIndent=-14,
        spaceAfter=4,
    )
    code = ParagraphStyle(
        "CodeCaption",
        parent=body,
        fontName=fonts["mono"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#111827"),
    )
    caption = ParagraphStyle(
        "Caption",
        parent=body,
        fontName=fonts["sans"],
        fontSize=9,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4b5563"),
        spaceBefore=4,
        spaceAfter=8,
    )
    return {
        "body": body,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "bullet": bullet,
        "number": number,
        "code": code,
        "caption": caption,
        "mono_name": fonts["mono"],
        "sans_name": fonts["sans"],
        "bold_name": fonts["bold"],
    }


def escape_text(text: str) -> str:
    return html.escape(text, quote=False)


def inline_markup(text: str, mono_font: str) -> str:
    escaped = escape_text(text)

    def repl(match: re.Match[str]) -> str:
        inner = html.escape(match.group(1), quote=False)
        return f'<font name="{mono_font}">{inner}</font>'

    return re.sub(r"`([^`]+)`", repl, escaped)


def resolve_image_path(md_path: Path, target: str) -> Path:
    candidate = (md_path.parent / target).resolve()
    if candidate.exists():
        return candidate
    repo_candidate = (ROOT / target).resolve()
    return repo_candidate


def scale_image(path: Path, max_width: float, max_height: float = 18 * cm) -> Image:
    img = Image(str(path))
    width = img.imageWidth
    height = img.imageHeight
    scale = min(max_width / width, max_height / height, 1.0)
    img.drawWidth = width * scale
    img.drawHeight = height * scale
    img.hAlign = "CENTER"
    return img


def parse_table_block(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if i == 1 and re.fullmatch(r"[\|\-\:\s]+", stripped):
            continue
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        cells = [cell.strip() for cell in stripped.split("|")]
        rows.append(cells)
    return rows


def render_table(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    col_count = max(len(row) for row in rows)
    padded = [row + [""] * (col_count - len(row)) for row in rows]
    data = []
    for row_idx, row in enumerate(padded):
        style = styles["h3"] if row_idx == 0 else styles["body"]
        data.append([Paragraph(inline_markup(cell, styles["mono_name"]), style) for cell in row])

    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), styles["sans_name"]),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def parse_markdown(md_path: Path, styles: dict[str, ParagraphStyle]):
    lines = md_path.read_text(encoding="utf-8").splitlines()
    story = []
    paragraph_buffer: list[str] = []
    i = 0

    def flush_paragraph() -> None:
        if not paragraph_buffer:
            return
        text = " ".join(part.strip() for part in paragraph_buffer).strip()
        if text:
            story.append(Paragraph(inline_markup(text, styles["mono_name"]), styles["body"]))
        paragraph_buffer.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            fence = stripped
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            story.append(
                Preformatted(
                    "\n".join(code_lines),
                    ParagraphStyle(
                        "CodeBlock",
                        fontName=styles["mono_name"],
                        fontSize=8.5,
                        leading=11,
                        leftIndent=10,
                        rightIndent=10,
                        borderPadding=8,
                        borderWidth=0.6,
                        borderColor=colors.HexColor("#d1d5db"),
                        backColor=colors.HexColor("#f3f4f6"),
                        spaceBefore=4,
                        spaceAfter=10,
                    ),
                )
            )
            i += 1
            continue

        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if image_match:
            flush_paragraph()
            alt_text, target = image_match.groups()
            image_path = resolve_image_path(md_path, target)
            if image_path.exists():
                story.append(scale_image(image_path, CONTENT_WIDTH))
                if alt_text:
                    story.append(Paragraph(escape_text(alt_text), styles["caption"]))
                else:
                    story.append(Spacer(1, 0.25 * cm))
            i += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            rows = parse_table_block(table_lines)
            if rows:
                story.append(render_table(rows, styles))
                story.append(Spacer(1, 0.25 * cm))
            continue

        if stripped.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(escape_text(stripped[2:]), styles["h1"]))
            i += 1
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(escape_text(stripped[3:]), styles["h2"]))
            i += 1
            continue

        if stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(escape_text(stripped[4:]), styles["h3"]))
            i += 1
            continue

        if re.match(r"^[-*] ", stripped):
            flush_paragraph()
            story.append(
                Paragraph(
                    inline_markup(stripped[2:].strip(), styles["mono_name"]),
                    styles["bullet"],
                    bulletText="•",
                )
            )
            i += 1
            continue

        ordered_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if ordered_match:
            flush_paragraph()
            num, content = ordered_match.groups()
            story.append(
                Paragraph(
                    inline_markup(content.strip(), styles["mono_name"]),
                    styles["number"],
                    bulletText=f"{num}.",
                )
            )
            i += 1
            continue

        if not stripped:
            flush_paragraph()
            story.append(Spacer(1, 0.12 * cm))
            i += 1
            continue

        paragraph_buffer.append(line)
        i += 1

    flush_paragraph()
    return story


def add_page_number(canvas, doc):
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(PAGE_WIDTH - MARGIN, 1.2 * cm, f"{canvas.getPageNumber()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Markdown to PDF")
    parser.add_argument("input_md", help="Input markdown path")
    parser.add_argument("output_pdf", nargs="?", help="Output PDF path")
    args = parser.parse_args()

    md_path = Path(args.input_md).resolve()
    pdf_path = Path(args.output_pdf).resolve() if args.output_pdf else md_path.with_suffix(".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    fonts = register_fonts()
    styles = build_styles(fonts)
    story = parse_markdown(md_path, styles)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=1.7 * cm,
        bottomMargin=1.6 * cm,
        title=md_path.stem,
        author="OpenAI Codex",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(pdf_path)


if __name__ == "__main__":
    main()
