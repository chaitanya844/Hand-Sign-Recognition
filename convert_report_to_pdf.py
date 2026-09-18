import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Preformatted
)

def md_to_pdf(md_path, pdf_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1a365d'),
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=16,
        textColor=colors.HexColor('#4a5568'),
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#2b6cb0'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2d3748'),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14.5,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#1a202c'),
        backColor=colors.HexColor('#edf2f7'),
        spaceBefore=6,
        spaceAfter=8,
        leftIndent=10,
        rightIndent=10
    )

    story = []
    lines = text.split('\n')
    i = 0
    total = len(lines)

    while i < total:
        line = lines[i]

        # Ignore empty lines
        if not line.strip():
            i += 1
            continue

        # Horizontal rule
        if line.strip() in ['---', '***', '___']:
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0'), spaceBefore=8, spaceAfter=10))
            i += 1
            continue

        # Code blocks
        if line.strip().startswith('```'):
            code_lines = []
            i += 1
            while i < total and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            if i < total:
                i += 1  # skip closing ```
            code_text = "\n".join(code_lines)
            story.append(Preformatted(code_text, code_style))
            continue

        # Markdown Table detection
        if '|' in line and i + 1 < total and re.match(r'^\s*\|?\s*[-:]+[-| :]*\|?\s*$', lines[i+1]):
            table_lines = []
            while i < total and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            
            # Parse table
            table_data = []
            for t_idx, tl in enumerate(table_lines):
                if t_idx == 1: # separator
                    continue
                cells = [c.strip() for c in tl.strip().strip('|').split('|')]
                row_cells = []
                for c in cells:
                    cell_p = Paragraph(f"<b>{c}</b>" if t_idx == 0 else c, body_style)
                    row_cells.append(cell_p)
                table_data.append(row_cells)

            if table_data:
                col_count = len(table_data[0])
                available_w = 504 # letter 612 - 108 margins
                col_w = available_w / col_count
                
                t = Table(table_data, colWidths=[col_w]*col_count)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf2f7')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1a202c')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
                    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#a0aec0')),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(t)
                story.append(Spacer(1, 8))
            continue

        # Title / Headings
        if line.startswith('# '):
            story.append(Paragraph(line[2:].strip(), title_style))
            i += 1
            continue
        elif line.startswith('## '):
            story.append(Paragraph(line[3:].strip(), h1_style))
            i += 1
            continue
        elif line.startswith('### '):
            story.append(Paragraph(line[4:].strip(), h2_style))
            i += 1
            continue

        # Bullet lists
        if re.match(r'^\s*[-*]\s+', line):
            content = re.sub(r'^\s*[-*]\s+', '', line).strip()
            # Convert markdown bold/italics
            content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
            content = re.sub(r'\*(.*?)\*', r'<i>\1</i>', content)
            content = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', content)
            story.append(Paragraph(f"&bull; {content}", bullet_style))
            i += 1
            continue

        # Numbered list
        num_match = re.match(r'^\s*(\d+)\.\s+', line)
        if num_match:
            content = line[num_match.end():].strip()
            content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
            content = re.sub(r'\*(.*?)\*', r'<i>\1</i>', content)
            content = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', content)
            story.append(Paragraph(f"{num_match.group(1)}. {content}", bullet_style))
            i += 1
            continue

        # Standard paragraph
        p_text = line.strip()
        p_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', p_text)
        p_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', p_text)
        p_text = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', p_text)
        story.append(Paragraph(p_text, body_style))
        i += 1

    doc.build(story)
    print(f"Successfully generated: {pdf_path}")

if __name__ == '__main__':
    md_file = r'c:\C Projects\computer vision\PROJECT REPORT.md'
    pdf_file = r'c:\C Projects\computer vision\PROJECT_REPORT.pdf'
    md_to_pdf(md_file, pdf_file)
