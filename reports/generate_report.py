"""Generate matching, numbered PDF and DOCX reports from actual experiment data.

Usage: python -m reports.generate_report
"""
import json
from pathlib import Path
from xml.sax.saxutils import escape

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, KeepTogether
import matplotlib

from config import ROOT
from reports.content import build_content
from src.export_figures import export_figures
from src.model_training import load_experiment

TITLE = "Concrete Strength Prediction Using Polynomial Regression"
SUBTITLE = "Statistical Machine Learning Project Report"


def clean(text):
    return str(text).replace('–','-').replace('—','-').replace('−','-').replace('’',"'").replace('\u00a0',' ')


def image_size(path,width):
    with PILImage.open(path) as im:
        return width,width*im.height/im.width


def contents(pages):
    return [(block[1],i+3) for i,page in enumerate(pages) for block in page
            if block[0]=='heading' and not block[1].endswith('continued')]


def pdf_report(path,pages,figures,meta):
    fonts=Path(matplotlib.get_data_path())/'fonts/ttf'
    for name,file in [('ReportSans','DejaVuSans.ttf'),('ReportSans-Bold','DejaVuSans-Bold.ttf')]:
        if name not in pdfmetrics.getRegisteredFontNames(): pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('ReportSans',normal='ReportSans',bold='ReportSans-Bold',italic='ReportSans',boldItalic='ReportSans-Bold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyReport',fontName='ReportSans',fontSize=10,leading=14.5,spaceAfter=9,textColor=colors.HexColor('#243247')))
    styles.add(ParagraphStyle(name='SectionReport',fontName='ReportSans-Bold',fontSize=16,leading=20,spaceBefore=8,spaceAfter=12,textColor=colors.HexColor('#14213D'),keepWithNext=True))
    styles.add(ParagraphStyle(name='CaptionReport',fontName='ReportSans',fontSize=8.3,leading=11,spaceBefore=3,spaceAfter=10,textColor=colors.HexColor('#526176')))
    styles.add(ParagraphStyle(name='EquationReport',fontName='ReportSans',fontSize=11.5,leading=18,spaceBefore=8,spaceAfter=12,alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='CellReport',fontName='ReportSans',fontSize=8.2,leading=11))
    styles.add(ParagraphStyle(name='CoverTitle',fontName='ReportSans-Bold',fontSize=28,leading=36,spaceAfter=26,textColor=colors.black))
    def para(text,style='BodyReport'): return Paragraph(escape(clean(text)),styles[style])
    story=[Spacer(1,65),para('STATISTICAL MACHINE LEARNING  /  MODULE 4','CaptionReport'),Spacer(1,24),
           para(TITLE,'CoverTitle'),para(SUBTITLE),Spacer(1,45),para('Bhoomika S','SectionReport'),
           para('Department of Artificial Intelligence and Data Science'),para('Don Bosco Institute of Technology, Bengaluru'),
           Spacer(1,35),para('SDG 9 - Industry, Innovation and Infrastructure'),
           para('An experimental comparison of linear and polynomial regression with cross-validation and statistical diagnostics.'),
           Spacer(1,35),para(f"Experiment {meta['experiment_id']}  |  Generated from execution dated {meta['created_utc'][:10]}",'CaptionReport'),PageBreak(),
           para('Contents','SectionReport')]
    toc_data=[[para(title,'CellReport'),para(str(number),'CellReport')] for title,number in contents(pages)]
    toc=Table(toc_data,colWidths=[425,40])
    toc.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('ALIGN',(-1,0),(-1,-1),'RIGHT')]))
    story.extend([toc,Spacer(1,20),para('Tables and figures are numbered consecutively. The original dataset, source code, split assignments and machine-readable results accompany this report.','CaptionReport')])
    for page in pages:
        story.append(PageBreak())
        for block in page:
            kind=block[0]
            if kind in ['heading','paragraph','equation']:
                story.append(para(block[1],{'heading':'SectionReport','paragraph':'BodyReport','equation':'EquationReport'}[kind]))
            elif kind=='figure':
                _,key,caption,width=block
                w,h=image_size(figures[key],width*72)
                story.append(KeepTogether([Image(str(figures[key]),width=w,height=h),para(caption,'CaptionReport')]))
            elif kind=='table':
                _,caption,frame=block
                rows=[[para(c,'CellReport') for c in frame.columns]]+[[para(v,'CellReport') for v in row] for row in frame.itertuples(index=False,name=None)]
                widths=[465/len(frame.columns)]*len(frame.columns)
                if len(frame.columns)==2: widths=[165,300]
                t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EEF7')),
                    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F8FAFC')]),
                    ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
                    ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#CBD5E1'))]))
                story.append(KeepTogether([para(caption,'CaptionReport'),t,Spacer(1,12)]))
    def footer(canvas,doc):
        canvas.setFont('ReportSans',8)
        canvas.setFillColor(colors.HexColor('#64748B'))
        canvas.drawString(57,29,'Bhoomika S  |  Concrete Strength Prediction')
        canvas.drawRightString(A4[0]-57,29,str(doc.page))
    document=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=57,leftMargin=57,topMargin=45,bottomMargin=48,
                               title=TITLE,author='Bhoomika S',subject=SUBTITLE)
    document.build(story,onFirstPage=footer,onLaterPages=footer)


def docx_report(path,pages,figures,meta):
    doc=Document()
    section=doc.sections[0]
    section.page_width,section.page_height=Inches(8.2677),Inches(11.6929)
    section.top_margin,section.bottom_margin=Inches(.64),Inches(.7)
    section.left_margin,section.right_margin=Inches(.79),Inches(.79)
    normal=doc.styles['Normal']
    normal.font.name='Arial'
    normal.font.size=Pt(10.5)
    normal.paragraph_format.line_spacing=1.12
    normal.paragraph_format.space_after=Pt(9)
    normal.paragraph_format.widow_control=True
    for name,size in [('Title',28),('Heading 1',16),('Heading 2',13)]:
        style=doc.styles[name]
        style.font.name='Arial'
        style.font.size=Pt(size)
        style.font.color.rgb=RGBColor.from_string('000000' if name=='Title' else '14213D')
        style.paragraph_format.space_after=Pt(12)
    doc.styles['Title'].font.bold=True
    # Remove inherited template paragraph rules; typography defines the cover.
    for border in doc.styles.element.xpath('.//w:pBdr'):
        border.getparent().remove(border)
    doc.styles['Caption'].font.size=Pt(8.5)
    doc.styles['Caption'].font.color.rgb=RGBColor.from_string('526176')
    footer=section.footer.paragraphs[0]
    footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run('Bhoomika S  |  Concrete Strength Prediction  |  ').font.size=Pt(8)
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    doc.core_properties.title=TITLE
    doc.core_properties.author='Bhoomika S'
    doc.core_properties.subject=SUBTITLE
    doc.add_paragraph('\n\nSTATISTICAL MACHINE LEARNING  /  MODULE 4')
    doc.add_paragraph(TITLE,'Title')
    doc.add_paragraph(SUBTITLE)
    author=doc.add_paragraph('\nBhoomika S')
    author.runs[0].bold=True
    doc.add_paragraph('Department of Artificial Intelligence and Data Science')
    doc.add_paragraph('Don Bosco Institute of Technology, Bengaluru')
    doc.add_paragraph('\nSDG 9 - Industry, Innovation and Infrastructure')
    doc.add_paragraph('An experimental comparison of linear and polynomial regression with cross-validation and statistical diagnostics.')
    doc.add_paragraph(f"\nExperiment {meta['experiment_id']}\nGenerated from execution dated {meta['created_utc'][:10]}")
    doc.add_page_break()
    doc.add_heading('Contents',level=1)
    for title,number in contents(pages):
        p=doc.add_paragraph()
        p.paragraph_format.space_after=Pt(7)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(6.2),WD_ALIGN_PARAGRAPH.RIGHT)
        p.add_run(f'{title}\t{number}')
    doc.add_paragraph('Tables and figures are numbered consecutively. Page numbers refer to this generated edition. Editing the text can change pagination.','Caption')
    for page in pages:
        doc.add_page_break()
        for block in page:
            kind=block[0]
            if kind=='heading': doc.add_heading(clean(block[1]),level=1)
            elif kind=='paragraph': doc.add_paragraph(clean(block[1]))
            elif kind=='equation':
                p=doc.add_paragraph(clean(block[1]));p.alignment=WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs: run.font.name='DejaVu Sans';run.font.size=Pt(11.5)
            elif kind=='figure':
                _,key,caption,width=block
                p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.keep_with_next=True
                run=p.add_run();shape=run.add_picture(str(figures[key]),width=Inches(width))
                shape._inline.docPr.set('descr',clean(caption))
                doc.add_paragraph(clean(caption),'Caption')
            elif kind=='table':
                _,caption,frame=block
                p=doc.add_paragraph(clean(caption),'Caption');p.paragraph_format.keep_with_next=True
                table=doc.add_table(rows=1,cols=len(frame.columns));table.style='Light Shading Accent 1'
                for cell,text in zip(table.rows[0].cells,frame.columns): cell.text=clean(text)
                repeat=OxmlElement('w:tblHeader');table.rows[0]._tr.get_or_add_trPr().append(repeat)
                for values in frame.itertuples(index=False,name=None):
                    for cell,text in zip(table.add_row().cells,values): cell.text=clean(text)
                if len(frame.columns)==2:
                    table.columns[0].width=Inches(2.3);table.columns[1].width=Inches(4.1)
                for row in table.rows:
                    cant=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(cant)
                    for cell in row.cells:
                        for p in cell.paragraphs:
                            p.paragraph_format.space_after=Pt(4);p.paragraph_format.space_before=Pt(4)
                            p.paragraph_format.line_spacing=1.0
                            for run in p.runs: run.font.size=Pt(8.5)
                doc.add_paragraph().paragraph_format.space_after=Pt(1)
    doc.save(path)


def generate_reports(bundle=None,output_dir=None):
    bundle=bundle if bundle is not None else load_experiment()
    meta=bundle['metadata']
    output=Path(output_dir) if output_dir else ROOT/'reports/output'/meta['experiment_id']
    output.mkdir(parents=True,exist_ok=True)
    figures=export_figures(bundle)
    pages,diagnostics=build_content(bundle)
    paths={kind:output/f'concrete_strength_report.{kind}' for kind in ['pdf','docx']}
    pdf_report(paths['pdf'],pages,figures,meta)
    docx_report(paths['docx'],pages,figures,meta)
    manifest={'experiment':meta,'metrics':bundle['metrics'].to_dict(orient='records'),
              'diagnostics':diagnostics['statistics'],'expected_pages':len(pages)+2,
              'contents':[{'heading':h,'page':p} for h,p in contents(pages)],
              'figures':{k:str(v.relative_to(ROOT)) for k,v in figures.items()}}
    (output/'report_manifest.json').write_text(json.dumps(manifest,indent=2))
    diagnostics['vif'].to_csv(output/'vif.csv',index=False)
    return paths


if __name__=='__main__':
    print({k:str(v) for k,v in generate_reports().items()})
