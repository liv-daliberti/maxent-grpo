#!/usr/bin/env python3
"""Add the Llama arm to the multi-family grid once an HF token is available.

meta-llama repositories are gated: accepting the licence on the website is
necessary but not sufficient, because the request must also authenticate as
that account. Run `huggingface-cli login` (or export HF_TOKEN) first; this
script then resolves the pinned revisions, registers the three scales,
downloads them, and extends the collection plan.
"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'var/cache/huggingface/transformers'
os.environ.setdefault('HF_HUB_CACHE', str(CACHE))
REPOS = [('llama1b', 'meta-llama/Llama-3.2-1B-Instruct'),
         ('llama3b', 'meta-llama/Llama-3.2-3B-Instruct'),
         ('llama8b', 'meta-llama/Llama-3.1-8B-Instruct')]
FIELDS = ('hidden_size', 'num_hidden_layers', 'num_attention_heads',
          'num_key_value_heads', 'intermediate_size', 'vocab_size')
TIME = {'llama1b': '03:00:00', 'llama3b': '04:00:00', 'llama8b': '06:00:00'}


def main() -> int:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi()
    try:
        who = api.whoami()
    except Exception as error:
        print('No Hugging Face token found. Run `huggingface-cli login` first.', file=sys.stderr)
        print(f'  ({type(error).__name__})', file=sys.stderr)
        return 2
    print(f'authenticated as {who.get("name")}')

    specs = {}
    for label, repo in REPOS:
        try:
            sha = api.model_info(repo).sha
            cfg = json.loads(Path(hf_hub_download(repo, 'config.json', revision=sha)).read_text())
        except Exception as error:
            print(f'{repo}: {type(error).__name__} -- licence not accepted for this account?',
                  file=sys.stderr)
            return 3
        specs[label] = (repo, sha, tuple(cfg[k] for k in FIELDS),
                        cfg['model_type'], cfg['architectures'])
        print(f'  {label}: {sha[:12]} {cfg["model_type"]}')

    source = ROOT / 'ops/evaluate_modebench_base_grid.py'
    text = source.read_text()
    if "'llama1b'" not in text:
        entries = ''.join(
            f"    '{k}': ('{v[0]}', '{v[1]}',\n"
            f"                {v[2]}, '{v[3]}', {v[4]}),\n" for k, v in specs.items())
        text = text.replace("}\n# Parameter counts", entries + "}\n# Parameter counts")
        text = text.replace("'olmo13b': 13.7,", "'olmo13b': 13.7, 'llama1b': 1.2, 'llama3b': 3.2, 'llama8b': 8.0,")
        text = text.replace("'olmo1b': 'OLMo-2',", "'llama1b': 'Llama-3', 'llama3b': 'Llama-3', 'llama8b': 'Llama-3',\n                'olmo1b': 'OLMo-2',")
        text = text.replace("'14b', 'qwen32b')", "'14b', 'qwen32b',\n                'llama1b', 'llama3b', 'llama8b')")
        source.write_text(text)
        print('registered three Llama scales in MODEL_SPECS')

    for label, repo in REPOS:
        snapshot_download(repo, revision=specs[label][1],
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'tokenizer*'])
        print(f'  downloaded {repo}')
    subprocess.run([sys.executable, str(ROOT / 'ops/extend_multifamily_plan.py'),
                    *specs], check=False)
    print('\nNext: rerun the plan builder, then submit the llama waves.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
