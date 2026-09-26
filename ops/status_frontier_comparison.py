#!/usr/bin/env python3
"""Read saved campaign state; never calls model APIs."""
import argparse,json,pathlib
from datetime import datetime,timezone
ROOT=pathlib.Path(__file__).resolve().parents[1]
RUNS=[('gpt-5.6-sol','gpt56sol'),('claude-opus-5','claude_opus5'),('claude-opus-4-8','claude_opus48'),('gpt-5.4','gpt54'),('grok-4.3','grok43'),('DeepSeek-V4-Pro','deepseek_v4_pro'),('FW-Kimi-K3','kimi_k3')]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--write',action='store_true');a=ap.parse_args()
 now=datetime.now(timezone.utc).isoformat();items=[]
 for model,slug in RUNS:
  directory=ROOT/'artifacts'/f'frontier_modebench_{slug}_20260911'
  p=directory/'status.json';s=json.loads(p.read_text()) if p.exists() else {}
  audit=directory/'completion_audit.json';report=directory/'summary.json'
  audited=json.loads(audit.read_text()).get('status') if audit.exists() else None
  summarized=json.loads(report.read_text()).get('status') if report.exists() else None
  items.append({'model':model,'directory':str(directory),'saved_responses':s.get('completed_samples',0),
   'expected_responses':15360,'response_status_counts':s.get('response_status_counts',{}),
   'collection_complete':s.get('complete',False),'completion_audit':audited,'summary':summarized,'status_updated_at_utc':s.get('updated_at_utc')})
 if a.write:
  out=ROOT/'artifacts/frontier_models_comparison_20260911'
  (out/'STATUS.json').write_text(json.dumps({'updated_at_utc':now,'models':items},indent=2)+'\n')
  lines=['# Hosted evaluation status','',f'Updated {now}. Counts come from durable run status files.','',
   '| Deployment | Saved / intended responses | Truncated | Integrity audit | Summary |','|---|---:|---:|---|---|']
  for r in items:
   lines.append(f"| {r['model']} | {r['saved_responses']:,} / 15,360 | {r['response_status_counts'].get('incomplete',0):,} | {r['completion_audit'] or 'pending'} | {r['summary'] or 'pending'} |")
  lines+=['','Each full cohort has 1,920 test prompts, five domains, three levels and eight stateless requests per prompt. No training. Raw responses, requests, grading receipts and attempts remain in the model directories.','',
  'Seven original-protocol model cohorts are tracked, including both Claude Opus 5 and Opus 4.8. Separate changed-prompt diagnostic cohorts are excluded from this inventory and the original-protocol comparison.','',
  '[Protocol](PROTOCOL.md) · [Completed-model comparison](COMPARISON.md)','']
  (out/'STATUS.md').write_text('\n'.join(lines))
 print(json.dumps({'updated_at_utc':now,'models':items}))
if __name__=='__main__':main()
