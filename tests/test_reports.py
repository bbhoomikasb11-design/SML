import json
from docx import Document
from pypdf import PdfReader
from reports.generate_report import generate_reports


def test_generate_reports_and_match_metrics(bundle,tmp_path):
    paths=generate_reports(bundle,tmp_path)
    manifest=json.loads((tmp_path/'report_manifest.json').read_text())
    assert manifest['metrics']==bundle['metrics'].to_dict(orient='records')
    reader=PdfReader(paths['pdf'])
    assert len(reader.pages)==manifest['expected_pages']
    text='\n'.join(page.extract_text() for page in reader.pages)
    for item in manifest['contents']:
        assert item['heading'] in reader.pages[item['page']-1].extract_text()
    selected=bundle['metrics'].set_index('Model').loc[bundle['metadata']['selected']]
    assert f"{selected['Test RMSE']:.3f}" in text
    assert 'Bhoomika S' in text and '22 References' in text
    document=Document(paths['docx'])
    assert len(document.inline_shapes)==11
    assert len(document.tables)==10
    word_text='\n'.join(p.text for p in document.paragraphs)
    assert f"{selected['Test RMSE']:.3f}" in word_text
    assert bundle['metadata']['experiment_id'] in word_text
