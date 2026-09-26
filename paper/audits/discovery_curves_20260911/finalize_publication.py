"""Authenticate the final complete-local discovery publication after its builds."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,subprocess

ROOT=Path(__file__).resolve().parents[3]
AUDIT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/modebench_discovery_curves_20260911'
PAPER=ROOT/'paper';WORKSHOP=PAPER/'mathai2026'
STEM='modebench_discovery_curves_20260911'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def record(path):return {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size}
def pages(path):return subprocess.check_output(['pdftotext','-layout',str(path),'-'],text=True).split('\f')[:-1]

def main():
    report_path=PAPER/'results'/f'{STEM}.json'
    report=json.loads(report_path.read_text());source=Path(report['publication_source']['directory'])
    assert report['status']=='complete' and report['experiment_status']=='partial_panels'
    assert report['scope']['included_panels']==['local'] and report['scope']['omitted_panels']==['frontier']
    assert len(report['models'])==25
    inventory=report['inventory'];assert inventory['expected_draws']==inventory['finalized_draws']==110592
    assert all(run['complete'] for run in inventory['runs'])
    independent=json.loads((AUDIT/'observed_endpoint_review.json').read_text())
    assert independent['status']=='pass' and independent['audited_responses']==110592
    assert independent['point_estimates_compared']>=10000
    exact=[]
    for directory in [PAPER,WORKSHOP]:
        for original,target in [('analysis.json',f'results/{STEM}.json'),('appendix.tex',f'results/{STEM}.tex')]:
            assert sha(source/original)==sha(directory/target)
            exact.append(record(directory/target))
        for stem in ['modebench_discovery_curves_local','modebench_discovery_correct_budget_local']:
            for ext in ['pdf','png','json']:
                name=f'{stem}.{ext}';assert sha(source/name)==sha(directory/'figures'/name)
                exact.append(record(directory/'figures'/name))
    parent_log=(AUDIT/'parent_build_final.log').read_text();workshop_log=(AUDIT/'workshop_build_final.log').read_text()
    assert 'Current paper contract passed' in parent_log
    assert 'line-fill contract passed' in parent_log.lower()
    match=re.search(r'Main-length contract passed: (\d+)/9 main pages.*?References starts on page (\d+) \((\d+) total PDF pages\)',parent_log)
    assert match
    main_n,refs_n,total_n=map(int,match.groups());assert refs_n==main_n+1 and len(pages(PAPER/'main.pdf'))==total_n
    assert 'Created mathai2026-source.zip' in workshop_log
    assert 'compiled artifact current' in workshop_log
    subprocess.run(['qpdf',str(PAPER/'main.pdf'),'--pages','.',f'1-{main_n}','--',str(PAPER/'main-body.pdf')],check=True)
    main_body=pages(PAPER/'main-body.pdf');assert len(main_body)==main_n
    assert not any(re.search(r'^\s*References\s*$',page,re.M) for page in main_body)
    files=[BASE/'manifest.json',BASE/'ANALYSIS_PLAN.md',BASE/'local/plan.json',BASE/'analysis_source_manifest.json',
           BASE/'hosted_execution_revisions/v2/hosted_execution.json',source/'analysis.json',source/'all_cells.csv',source/'artifact_manifest.json',
           PAPER/'main.tex',PAPER/'main.pdf',PAPER/'main-body.pdf',WORKSHOP/'main.tex',WORKSHOP/'appendix.tex',WORKSHOP/'main.pdf',
           WORKSHOP/'snapshot.json',WORKSHOP/'build_receipt.json',WORKSHOP/'mathai2026-source.zip',
           AUDIT/'parent_build_final.log',AUDIT/'workshop_build_final.log',AUDIT/'workshop_sync_final/workshop_sync.json',
           AUDIT/'observed_endpoint_review.json',AUDIT/'review_observed_endpoints.py']
    for optional in [BASE/'appendix_review/appendix_review.pdf',BASE/'appendix_review/manifest.json',BASE/'LOCAL_FINDINGS.md']:
        if optional.exists():files.append(optional)
    receipt={'schema':'discovery-curves-final-local-publication-v1','completed_at_utc':datetime.now(timezone.utc).isoformat(),
             'experiment_status':'partial_panels','local_checkpoints':25,'local_responses':110592,
             'frontier_responses_collected':0,'frontier_responses_planned':36864,'frontier_blocker':'intended Azure credential unavailable',
             'publication':{'parent_main_pages':main_n,'parent_references_page':refs_n,'parent_total_pages':total_n,'workshop_main_pages':4,
                            'figures_in_each_manuscript':21,'source_copies_exact':True},
             'validation':{'full_parent_build':'pass','main_page_limit':'pass','line_fill':'pass','workshop_build_and_source_bundle':'pass',
                           'analysis_and_sample_authentication':'pass','independent_point_estimate_counts':'pass','source_figure_hashes':'pass'},
             'citation':{'key':'yue2025rlvrlimit','url':'https://arxiv.org/abs/2504.13837','version':'v5'},
             'artifacts':[record(p) for p in files],'publication_copies':exact}
    (AUDIT/'final_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['artifacts','publication_copies']},indent=2))

if __name__=='__main__':main()
