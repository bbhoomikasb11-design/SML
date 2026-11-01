"""Rebuild the baseline dataset audit, models, diagnostics, figures and reports."""
import json
from config import ROOT
from src.model_training import run_experiment
from src.diagnostics import calculate_diagnostics
from src.feature_analysis import associations
from reports.generate_report import generate_reports
from src.eda import export_interactive


def main():
    bundle=run_experiment()
    diagnostic=calculate_diagnostics(bundle)
    output=ROOT/'artifacts/metrics'
    (output/'diagnostics.json').write_text(json.dumps(diagnostic['statistics'],indent=2))
    diagnostic['vif'].to_csv(output/'vif.csv',index=False)
    diagnostic['unusual'].to_csv(output/'largest_errors.csv',index_label='source_row')
    associations(bundle['data']).to_csv(output/'feature_associations.csv',index=False)
    reports=generate_reports(bundle)
    export_interactive(bundle)
    print(bundle['metrics'].round(4).to_string(index=False))
    print('Selected:',bundle['metadata']['selected'])
    for kind,path in reports.items(): print(kind.upper(),path)


if __name__=='__main__': main()
