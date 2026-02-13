"""
PDF Exporter
Generates professional PDF reports from analysis data
"""

import io
from datetime import datetime
from typing import Dict, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


class PDFExporter:
    """
    PDF Exporter for generating professional trading analysis reports.
    """
    
    def __init__(self, page_size: Tuple = A4):
        """
        Initialize PDF Exporter.
        
        Args:
            page_size: Page size tuple (default A4)
        """
        self.page_size = page_size
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a1a2e'),
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#16213e'),
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            leading=14,
        ))
        
        # Score style
        self.styles.add(ParagraphStyle(
            name='ScoreStyle',
            parent=self.styles['Normal'],
            fontSize=36,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#0f3460'),
            spaceAfter=10,
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            textColor=colors.grey,
            spaceAfter=20,
        ))
    
    def export(
        self,
        technical_summary: Dict,
        orderbook_summary: Dict,
        final_score: Dict,
        ai_report: Optional[str] = None,
        filename: Optional[str] = None
    ) -> bytes:
        """
        Generate PDF report.
        
        Args:
            technical_summary: Technical analysis summary
            orderbook_summary: Orderbook analysis summary
            final_score: Final score summary
            ai_report: Optional AI-generated report text
            filename: Optional filename (not used when returning bytes)
            
        Returns:
            PDF content as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self.page_size,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Build content
        elements = []
        
        # Title
        elements.append(Paragraph(
            "Crypto Trading Analysis Report",
            self.styles['CustomTitle']
        ))
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        elements.append(Paragraph(
            f"Generated: {timestamp}",
            self.styles['Subtitle']
        ))
        
        # Horizontal line
        elements.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor('#e0e0e0'),
            spaceAfter=20
        ))
        
        # Score Summary Section
        elements.extend(self._build_score_section(final_score))
        
        # Technical Analysis Section
        elements.extend(self._build_technical_section(technical_summary))
        
        # Orderbook Analysis Section
        elements.extend(self._build_orderbook_section(orderbook_summary))
        
        # AI Report Section (if available)
        if ai_report:
            elements.extend(self._build_ai_section(ai_report))
        
        # Disclaimer
        elements.extend(self._build_disclaimer())
        
        # Build PDF
        doc.build(elements)
        
        # Get bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def _build_score_section(self, final_score: Dict) -> list:
        """Build the score summary section."""
        elements = []
        
        elements.append(Paragraph(
            "OVERALL SCORE",
            self.styles['SectionHeader']
        ))
        
        # Main score display
        score_value = final_score.get('final_score', 0)
        score_color = self._get_score_color(score_value)
        
        elements.append(Paragraph(
            f"<font color='{score_color}'>{score_value}/10</font>",
            self.styles['ScoreStyle']
        ))
        
        # Signal and Risk Grade
        signal = final_score.get('signal', 'N/A')
        risk_grade = final_score.get('risk_grade', 'N/A')
        confidence = final_score.get('confidence', 'N/A')
        
        # Summary table
        summary_data = [
            ['Signal', 'Risk Grade', 'Confidence'],
            [signal, risk_grade, confidence],
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f0f0f0')),
            ('FONTSIZE', (0, 1), (-1, 1), 12),
            ('TOPPADDING', (0, 1), (-1, 1), 10),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 20))
        
        # Recommendation
        recommendation = final_score.get('recommendation', '')
        if recommendation:
            elements.append(Paragraph(
                "<b>Recommendation:</b>",
                self.styles['BodyText']
            ))
            for line in recommendation.split('\n'):
                if line.strip():
                    elements.append(Paragraph(line.strip(), self.styles['BodyText']))
        
        return elements
    
    def _build_technical_section(self, technical_summary: Dict) -> list:
        """Build the technical analysis section."""
        elements = []
        
        elements.append(Paragraph(
            "TECHNICAL ANALYSIS",
            self.styles['SectionHeader']
        ))
        
        # Technical score
        tech_score = technical_summary.get('overall_score', 0)
        signal = technical_summary.get('signal', 'N/A')
        confidence = technical_summary.get('confidence_percent', 0)
        interpretation = technical_summary.get('interpretation', 'N/A')
        
        elements.append(Paragraph(
            f"<b>Score:</b> {tech_score}/10 | <b>Signal:</b> {signal} | <b>Confidence:</b> {confidence}%",
            self.styles['BodyText']
        ))
        
        elements.append(Paragraph(
            f"<b>Analysis:</b> {interpretation}",
            self.styles['BodyText']
        ))
        
        # Component scores table
        component_scores = technical_summary.get('component_scores', {})
        if component_scores:
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("<b>Indicator Scores:</b>", self.styles['BodyText']))
            
            score_data = [['Indicator', 'Score']]
            for indicator, score in component_scores.items():
                score_display = f"{score:.1f}" if score is not None else "N/A"
                score_data.append([indicator, score_display])
            
            score_table = Table(score_data, colWidths=[3*inch, 2*inch])
            score_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d3436')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ]))
            
            elements.append(score_table)
        
        return elements
    
    def _build_orderbook_section(self, orderbook_summary: Dict) -> list:
        """Build the orderbook analysis section."""
        elements = []
        
        elements.append(Paragraph(
            "ORDERBOOK ANALYSIS",
            self.styles['SectionHeader']
        ))
        
        # Orderbook score
        ob_score = orderbook_summary.get('overall_score', 0)
        signal = orderbook_summary.get('signal', 'N/A')
        confidence = orderbook_summary.get('confidence_percent', 0)
        interpretation = orderbook_summary.get('interpretation', 'N/A')
        
        elements.append(Paragraph(
            f"<b>Score:</b> {ob_score}/10 | <b>Signal:</b> {signal} | <b>Confidence:</b> {confidence}%",
            self.styles['BodyText']
        ))
        
        elements.append(Paragraph(
            f"<b>Analysis:</b> {interpretation}",
            self.styles['BodyText']
        ))
        
        # Metrics table
        metrics = orderbook_summary.get('metrics', {})
        if metrics:
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("<b>Orderbook Metrics:</b>", self.styles['BodyText']))
            
            metrics_data = [
                ['Metric', 'Value'],
                ['Total Bid Volume', f"{metrics.get('total_bid_volume', 0):,.2f}"],
                ['Total Ask Volume', f"{metrics.get('total_ask_volume', 0):,.2f}"],
                ['Imbalance Ratio', f"{metrics.get('imbalance_ratio', 0):.4f}"],
                ['Spread %', f"{metrics.get('spread_percent', 0):.4f}%"],
                ['Buy Walls', str(metrics.get('buy_walls_count', 0))],
                ['Sell Walls', str(metrics.get('sell_walls_count', 0))],
            ]
            
            metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d3436')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ]))
            
            elements.append(metrics_table)
        
        return elements
    
    def _build_ai_section(self, ai_report: str) -> list:
        """Build the AI analysis section."""
        elements = []
        
        elements.append(Paragraph(
            "AI ANALYSIS",
            self.styles['SectionHeader']
        ))
        
        # Parse markdown-style sections
        paragraphs = ai_report.split('\n')
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Handle markdown headers
            if para.startswith('**') and para.endswith('**'):
                para = f"<b>{para[2:-2]}</b>"
            elif para.startswith('# '):
                para = f"<b>{para[2:]}</b>"
            elif para.startswith('## '):
                para = f"<b>{para[3:]}</b>"
            
            # Handle bullet points
            if para.startswith('- '):
                para = f"• {para[2:]}"
            
            elements.append(Paragraph(para, self.styles['BodyText']))
        
        return elements
    
    def _build_disclaimer(self) -> list:
        """Build the disclaimer section."""
        elements = []
        
        elements.append(Spacer(1, 30))
        elements.append(HRFlowable(
            width="100%",
            thickness=0.5,
            color=colors.HexColor('#cccccc'),
            spaceAfter=10
        ))
        
        disclaimer_text = """
        <b>DISCLAIMER:</b> This analysis is generated automatically from screenshot data and AI interpretation. 
        It is for informational purposes only and should not be considered financial advice. 
        Cryptocurrency trading involves significant risk. Always conduct your own research and 
        consult with a qualified financial advisor before making investment decisions. 
        Past performance does not guarantee future results.
        """
        
        elements.append(Paragraph(
            disclaimer_text,
            ParagraphStyle(
                'Disclaimer',
                parent=self.styles['Normal'],
                fontSize=8,
                textColor=colors.grey,
                leading=10,
            )
        ))
        
        return elements
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on score value."""
        if score >= 7:
            return '#27ae60'  # Green
        elif score >= 5:
            return '#f39c12'  # Orange
        elif score >= 3:
            return '#e67e22'  # Dark orange
        else:
            return '#e74c3c'  # Red


def export_pdf(
    technical_summary: Dict,
    orderbook_summary: Dict,
    final_score: Dict,
    ai_report: Optional[str] = None
) -> bytes:
    """
    Convenience function to export PDF.
    
    Args:
        technical_summary: Technical analysis summary
        orderbook_summary: Orderbook analysis summary
        final_score: Final score summary
        ai_report: Optional AI report text
        
    Returns:
        PDF content as bytes
    """
    exporter = PDFExporter()
    return exporter.export(
        technical_summary,
        orderbook_summary,
        final_score,
        ai_report
    )
