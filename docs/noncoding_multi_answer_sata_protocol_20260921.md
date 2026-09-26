# Reuters topic tagging: noncoding pilot contract

The primary noncoding application is **select one relevant Reuters news topic** from the human-validated SATA-Bench menu. Each document has multiple accepted native taxonomy labels. A sample returns one letter; its mode is the underlying topic, never the letter. Repeated samples should cover different valid topics while retaining annotation agreement. This is an explicit one-answer adaptation of SATA's original select-all task, not an official SATA result. The original select-all target alone would have one valid set and would not establish multiple output modes.

## Sources and scope

- [SATA-Bench paper](https://arxiv.org/abs/2506.00643), Xu et al. (2025): real source documents, sampled taxonomy distractors, and final correction/consensus filtering by three human annotators. NEWS derives from Reuters-21578. No LLM-generated passages, questions, or distractor text are added here.
- [Canonical released dataset](https://huggingface.co/datasets/sata-bench/sata-bench/tree/ba43a7ab537adfa3498e3a160a6d1eafbefc95c1): `data_main.json`, 1,604 rows; dataset card specifies **CC-BY-NC-4.0**. Its direct answer/distractor groups are authoritative for this pilot.
- [Original Reuters-21578 source](https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection): human newswire articles and topic labels; UCI reports CC-BY-4.0. Preserve the SATA noncommercial restriction for its curated release.
- The GitHub helper JSON contains 1,650 rows and stale letter-answer fields. It is retained only as source-comparison evidence; neither those letters nor its labels supply our gold targets. The GitHub MIT software license is not substituted for the HF dataset license.

## Frozen materialization

`var/artifacts/noncoding_multi_answer_sata_20260921/` contains raw downloads, manifest, adapter config, records, and a train/dev review packet. The canonical raw SHA256 is `d9809889057a6f37bf0dd35b371a4cf3f3a23ff431f9fc69201220598e950112`; the records SHA256 is `de530462b2f3e9769cef9332103ca3fcf578725d590676cb6a4e22594de6cb54`.

There are 243 retained documents: **128 train, 32 dev, 83 reserved test**. Five of the release's 248 NEWS rows have extra nonwhitespace material in the question field and are excluded by a fixed structural rule; their exact fields and indices are recorded. All 243 retain the original paragraph, question, positive labels and distractor labels byte-for-byte. No labels are relabeled or silently dropped. Each menu has six options, with two to five positive topics.

Documents are grouped using normalized content and character 5-shingle Jaccard similarity >=0.8 before splitting; this check found no qualifying duplicate pairs. Source-selection examples already inspected are forced to train and listed in the manifest. Remaining groups are ordered by a fixed SHA256 rule. Test is fixed before model runs and cannot be loaded by the adapter without explicit `allow_test=true`. Structural metadata checks do not inspect test predictions.

Option order is fixed once: a SHA256 permutation followed by a rotation that balances gold positions across each split. The training gold-position counts for A–F are 49,49,48,48,49,49. Permutations never change within a prompt or between paired methods. These labels define an annotation-based reference set; they do not assert exhaustive truth about every possible topic a reader might consider relevant. Reuters parent/child categories, such as grain and wheat, are distinct taxonomy tags, not paraphrases or arbitrary letter variants.

## Adapter and verification

Use `oat_drgrpo.noncoding_multi_answer_sata.load_tasks(config)` with the generated `noncoding_multi_answer_sata_adapter_config.json`. It returns the common task interface used by the HF trainer and capability runner: `task_id`, complete Qwen2.5 ChatML `prompt`, `family`, `split`, `metadata`, and `verify(text)`. `metadata.known_mode_count` is the released positive-topic count.

Verification strips surrounding whitespace and accepts exactly one uppercase ASCII letter from the menu iff its native topic is source-positive. Lists, lowercase letters, prose, unknown letters and negative labels receive zero acceptance and no mode key. Ordinary wrong answers have `hard_violations=[]`; rejection reasons are in the receipt. Schema/hash/annotation corruption raises an integrity error. No semantic or LLM judge runs during training or evaluation.

Replay must store one exemplar per `(task_id,native_topic)` and use the same rendered prompt for generation and scoring. Prompt text does not expose gold labels or their count. The paired base model is Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`, as selected by the root experiment runner.

## Evaluation interpretation

Report annotation acceptance, native-topic coverage divided by known support, accepted-topic entropy/effective support, and missing annotated-topic mass where probabilities are available. Report per-prompt averages with uncertainty over prompts/seeds. Compare matched sample budgets and the same temperature/top-p/max-output settings. The task tests discovery and retention of relevant document tags; it does not demonstrate free-form reasoning diversity or official select-all accuracy. A paper-ready result still needs the paired model runs, variance assessment and a final untouched-test evaluation after the protocol is fixed.

## Data choices rejected

Original MultiRC is human-authored but the inspected records contain conflicting paraphrase labels (e.g. a one-word topic true while its expanded equivalent is false) and many synonymous positive options. The source, SuperGLUE zip and HF mirror agree on these problems. Its prototype adapter/review queue are not the ready dataset. BIOMED is deferred because a heart-failure article has Diseases among its negative choices, exposing an indexing-versus-semantic-relevance ambiguity. Neither domain is used to pad this pilot's sample count.

The current source evidence and review packet are assistant-assisted audits, not new human annotations. The final paper should attribute human labeling only to the original Reuters/SATA creators.

## Reproduction

```bash
PYTHONPATH=src python ops/noncoding_multi_answer_sata_prepare.py
PYTHONPATH=src python -m unittest discover -s tests -p 'test_noncoding_multi_answer_sata.py'
```

The builder is CPU-only, reads the pinned local source bytes, and emits the exact frozen records. The source and test boundary must not be changed based on model success.

```bibtex
@misc{xu2025satabench,
  title={SATA-Bench: Select All That Apply Benchmark for Multiple Choice Questions},
  author={Xu, Weijie and Cui, Shixian and Fang, Xi and Xue, Chi and Eckman, Stephanie and Reddy, Chandan K.},
  year={2025},
  eprint={2506.00643},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2506.00643}
}
```
