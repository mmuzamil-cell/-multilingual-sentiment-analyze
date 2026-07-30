import csv
import io
import pandas as pd
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from config import EXPORT_DIR, get_logger

logger = get_logger("report_generator")

def generate_csv_report(df: pd.DataFrame) -> bytes:
    """Generates a CSV report from predictions dataframe and returns bytes."""
    output = io.StringIO()
    # Ensure correct columns and format
    cols_to_export = ["id", "review", "language", "prediction", "confidence", "timestamp"]
    available_cols = [c for c in cols_to_export if c in df.columns]
    
    df[available_cols].to_csv(output, index=False, encoding="utf-8")
    return output.getvalue().encode("utf-8")

def generate_pdf_report(df: pd.DataFrame, stats: dict = None) -> bytes:
    """
    Generates a professional PDF report summarizing the predictions.
    Returns the PDF as raw bytes.
    """
    logger.info("Generating PDF report...")
    pdf_buffer = io.BytesIO()
    
    # 1. Initialize Document
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    # 2. Setup Styles
    styles = getSampleStyleSheet()
    
    # Custom colors
    primary_color = colors.HexColor("#1e3d59")
    secondary_color = colors.HexColor("#17b978")
    dark_neutral = colors.HexColor("#222831")
    light_neutral = colors.HexColor("#f5f5f5")
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        textColor=primary_color,
        leading=28,
        alignment=0, # Left-aligned
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        textColor=colors.HexColor("#666666"),
        leading=16,
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=primary_color,
        leading=20,
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        textColor=dark_neutral,
        leading=14,
        spaceAfter=8
    )
    
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=11,
        textColor=dark_neutral
    )
    
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
        textColor=colors.white
    )

    story = []
    
    # --- HEADER / TITLE ---
    story.append(Paragraph("Smart Multilingual Product Review Analysis", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} — Sentiment Classification Report", subtitle_style))
    story.append(Spacer(1, 10))
    
    # --- STATISTICS SECTION ---
    story.append(Paragraph("Executive Summary", heading_style))
    
    if stats:
        summary_text = (
            f"This report presents the analysis of <b>{stats.get('total', len(df))}</b> product reviews processed by the "
            f"Smart Multilingual Sentiment Analysis system. The system recorded an average classification confidence score of "
            f"<b>{stats.get('avg_confidence', 0.0) * 100:.2f}%</b>. Below is the sentiment distribution breakdown:"
        )
    else:
        summary_text = f"This report lists the sentiment analysis predictions for {len(df)} review(s) processed by the system."
        
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 10))
    
    # Statistics Table
    if stats:
        stats_data = [
            [
                Paragraph("<b>Metric</b>", table_cell_style), 
                Paragraph("<b>Value / Distribution</b>", table_cell_style)
            ],
            [
                Paragraph("Total Reviews Analyzed", table_cell_style), 
                Paragraph(str(stats.get("total", 0)), table_cell_style)
            ],
            [
                Paragraph("Average Prediction Confidence", table_cell_style), 
                Paragraph(f"{stats.get('avg_confidence', 0.0) * 100:.2f}%", table_cell_style)
            ],
            [
                Paragraph("Positive Sentiment Ratio", table_cell_style), 
                Paragraph(f"{stats.get('positive_ratio', 0.0) * 100:.2f}%", table_cell_style)
            ],
            [
                Paragraph("Neutral Sentiment Ratio", table_cell_style), 
                Paragraph(f"{stats.get('neutral_ratio', 0.0) * 100:.2f}%", table_cell_style)
            ],
            [
                Paragraph("Negative Sentiment Ratio", table_cell_style), 
                Paragraph(f"{stats.get('negative_ratio', 0.0) * 100:.2f}%", table_cell_style)
            ]
        ]
        
        stats_table = Table(stats_data, colWidths=[2.5*inch, 3.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), primary_color),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_neutral]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#dddddd")),
        ]))
        # Overwrite text colors in first header row of table data
        for col_idx in range(len(stats_data[0])):
            stats_data[0][col_idx].style.textColor = colors.white
            
        story.append(stats_table)
        story.append(Spacer(1, 20))
        
    # --- PREDICTIONS TABLE ---
    story.append(Paragraph("Detailed Review Logs", heading_style))
    story.append(Paragraph("Below is a log of the most recent sentiment predictions:", body_style))
    story.append(Spacer(1, 10))
    
    # Format predictions table
    # Columns: Timestamp, Review (Truncated), Language, Prediction, Confidence
    table_data = [
        [
            Paragraph("Timestamp", table_header_style),
            Paragraph("Review Text", table_header_style),
            Paragraph("Language", table_header_style),
            Paragraph("Sentiment", table_header_style),
            Paragraph("Conf.", table_header_style)
        ]
    ]
    
    # Sort or limit df to top 50 to avoid massive PDF sizes in preview
    df_sorted = df.copy()
    if len(df_sorted) > 50:
        df_sorted = df_sorted.head(50)
        
    for _, row in df_sorted.iterrows():
        # Format Timestamp
        ts = row.get("timestamp", "")
        if isinstance(ts, str):
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                ts_formatted = dt.strftime("%b %d, %H:%M")
            except Exception:
                ts_formatted = ts[:16]
        else:
            ts_formatted = str(ts)
            
        # Truncate review text
        review = str(row.get("review", ""))
        if len(review) > 75:
            review_trunc = review[:72] + "..."
        else:
            review_trunc = review
            
        lang = str(row.get("language", "")).upper()
        pred = str(row.get("prediction", ""))
        conf = f"{float(row.get('confidence', 0.0)) * 100:.1f}%"
        
        # Color code prediction text in PDF
        pred_color = "#2ecc71" if pred == "Positive" else ("#e74c3c" if pred == "Negative" else "#f39c12")
        pred_p = Paragraph(f'<font color="{pred_color}"><b>{pred}</b></font>', table_cell_style)
        
        table_data.append([
            Paragraph(ts_formatted, table_cell_style),
            Paragraph(review_trunc, table_cell_style),
            Paragraph(lang, table_cell_style),
            pred_p,
            Paragraph(conf, table_cell_style)
        ])
        
    # Column widths (Total page width is 8.5in - 2*0.75in margin = 7.0in)
    col_widths = [1.2*inch, 3.2*inch, 0.8*inch, 1.1*inch, 0.7*inch]
    
    pred_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_neutral]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#dddddd")),
    ]))
    
    story.append(pred_table)
    
    # If truncated list, add note
    if len(df) > 50:
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"* This report lists only the 50 most recent records out of {len(df)} total reviews.", ParagraphStyle("Note", parent=styles["Normal"], fontSize=8, textColor=colors.gray)))
        
    # 4. Build Document
    doc.build(story)
    
    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()
    return pdf_bytes
