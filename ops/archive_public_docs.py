"""Pure rendering of archive documentation. No filesystem, network, or model operations."""
from collections import Counter
import re

MODELS = {'qwen05b': 'Qwen2.5-0.5B-Instruct', 'falcon1b': 'Falcon3-1B-Instruct', 'qwen3b': 'Qwen2.5-3B-Instruct'}
DOMAINS = {'countdown': 'Countdown', 'graph_coloring': 'Graph Coloring', 'python_factors': 'Python Factors', 'mathir': 'MathIR', 'pantry_plan': 'PantryPlan'}
METHODS = {'control': ('Dr.GRPO', 'Compute-matched control with an exactly zero replay derivative.'),
           'replay': ('Re:Dr.GRPO', 'Dr.GRPO with uniform likelihood rehearsal over retained verified canonical keys.'),
           'drgrpo': ('Dr.GRPO', 'Fresh Dr.GRPO objective; compute-matched bank traversal with zero replay derivative.'),
           'replay_drgrpo': ('Re:Dr.GRPO', 'Dr.GRPO plus verified canonical replay.'),
           'maxrl': ('MaxRL', 'Binary MaxRL objective on fresh rollouts; zero replay derivative.'),
           'replay_maxrl': ('Re:MaxRL', 'The same binary MaxRL objective plus verified canonical replay.'),
           'fresh_frequency': ('Re:Dr.GRPO · fresh-frequency', 'Replay weights proportional to validator-positive fresh-rollout counts for retained keys.')}
STUDIES = {
    'e78': {'name': 'E78', 'title': 'Verified replay · Qwen2.5-0.5B', 'registered': 50,
            'question': 'Does replay of the policy’s own verified exemplars preserve correct output support beyond compute-matched Dr.GRPO?',
            'design': 'Level 1 · five domains · five seeds (43–47) · Dr.GRPO and Re:Dr.GRPO. The replay intervention is uniform teacher-forced likelihood over one retained exemplar per discovered verified canonical key.',
            'paper': 'Experiment 1: retention across model scales. These are also the Qwen2.5-0.5B Dr.GRPO comparators for the Level-1 factorial.'},
    'e79': {'name': 'E79', 'title': 'Verified replay · Falcon3-1B', 'registered': 50,
            'question': 'Does the matched Dr.GRPO versus verified-replay comparison replicate with Falcon3-1B?',
            'design': 'Level 1 · five domains · five seeds (55–59) · Dr.GRPO and Re:Dr.GRPO. The two arms share the aligned Falcon recipe; only the replay derivative differs.',
            'paper': 'Experiment 1: retention across model scales. These are also the Falcon3-1B Dr.GRPO comparators for the Level-1 factorial. The archival selection omits the inadmissible Countdown replay seed 59 source; it is not replaced or imputed.'},
    'e80r1': {'name': 'E80-R1', 'title': 'Verified replay · Qwen2.5-3B', 'registered': 50,
              'question': 'Does the matched Dr.GRPO versus verified-replay comparison replicate with Qwen2.5-3B?',
              'design': 'Level 1 · five domains · five seeds (70–74) · Dr.GRPO and Re:Dr.GRPO. E80-R1 is the corrected replication and retains its own source identity; the superseded E80 is not substituted.',
              'paper': 'Experiment 1: retention across model scales. These are also the Qwen2.5-3B Dr.GRPO comparators for the Level-1 factorial.'},
    'e118': {'name': 'E118', 'title': 'MaxRL × verified replay · Level 1', 'registered': 150,
             'question': 'Does verified canonical replay add value beyond changing the fresh-rollout objective to binary MaxRL?',
             'design': 'Three base models · five Level-1 domains · five seeds per model · MaxRL and Re:MaxRL. These 150 registered cells supply the MaxRL side of a four-arm factorial with the separately archived Dr.GRPO pairs.',
             'paper': 'Experiment 2: the objective-by-replay factorial. Compare Re:MaxRL minus MaxRL within a model, domain and seed; compare that with Re:Dr.GRPO minus Dr.GRPO from E78, E79 or E80-R1 on the common admitted seed intersection.'},
    'e119': {'name': 'E119', 'title': 'Four-method factorial · Level 2', 'registered': 100,
             'question': 'How do the task objective and verified replay interact on the matched, structurally harder Level-2 benchmark?',
             'design': 'Qwen2.5-0.5B · five Level-2 domains · five seeds (43–47) · Dr.GRPO, Re:Dr.GRPO, MaxRL and Re:MaxRL. All four arms are newly trained on Level 2; Level-1 endpoints are not reused as Level-2 treatment comparators.',
             'paper': 'Experiment 3: matched levels. Within Level 2, estimate both within-objective replay contrasts and their interaction. Cross-level comparisons use different prompt sets and retain their separate level labels.'},
    'e120r1': {'name': 'E120-R1', 'title': 'Uniform versus fresh-frequency replay', 'registered': 45,
               'question': 'Does uniform weighting over discovered canonical keys preserve broader verified support than weighting those keys by their fresh-rollout frequency?',
               'design': '45 fresh-frequency treatment cells: all five Qwen2.5-0.5B domains (25), plus Graph Coloring and PantryPlan for Falcon3-1B (10) and Qwen2.5-3B (10). Match each treatment to its existing uniform-replay source by model, domain and seed.',
               'paper': 'The registered frequency-weighting ablation. Its primary frozen analysis uses the 25 Qwen2.5-0.5B pairs across five domains; larger-model exports are extensions, not additions to that frozen primary estimate. The contrast is uniform minus fresh-frequency for correctness and extra verified modes.'},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def anchor(name):
    return f'<a id="{name}"></a>'


def landing(exp):
    return f'experiments/{exp}/README.md'


def _table(repo, rows, methods_registry=METHODS):
    if not rows:
        return 'Transfer in progress. No verified model folders are available in this subset yet.'
    lines = ['| Model | Method | Seed | Terminal export | Immutable revision |', '| --- | --- | ---: | --- | --- |']
    for row in sorted(rows, key=lambda r: (r['model'], r['method'], r['seed'])):
        url = f"https://huggingface.co/{repo}/tree/{row['commit_sha']}/{row['repo_prefix']}"
        lines.append(f"| {row['model']} | {methods_registry[row['method']][0]} | {row['seed']} | [step {row['step']}]({url}) | [`{row['commit_sha'][:12]}`]({url}) |")
    return '\n'.join(lines)


def render_readmes(plan, catalog, *, include_guides=False, additional_studies=None, additional_methods=None, available_sections=None, benchmark_revision=None):
    """Return root + six study READMEs from the approved selection and verified rows.

    The caller owns publication. Only verified catalog rows produce model-folder
    links. Plan rows determine coverage and original source identities, never a
    claim of remote availability. The final builder can add its separate guides.
    """
    studies = {**STUDIES, **(additional_studies or {})}
    methods_registry = {**METHODS, **(additional_methods or {})}
    require(not (set(additional_studies or {}) & set(STUDIES)), 'Existing study definitions cannot be replaced')
    require(not (set(additional_methods or {}) & set(METHODS)), 'Existing method definitions cannot be replaced')
    require(benchmark_revision is None or re.fullmatch(r'[0-9a-f]{40}', benchmark_revision), 'Unpinned benchmark revision')
    benchmark_nav = ([f'[Benchmarks](https://huggingface.co/datasets/od2961/ModeBench/blob/{benchmark_revision}/README.md)'] if benchmark_revision else [])
    sections = dict(plan.get('publication_sections') or {})
    sections.update(available_sections or {})
    require(set(sections) <= {'Data', 'Results', 'Reproducibility'}, 'Unknown archive navigation section')
    for value in sections.values():
        require(re.fullmatch(r'[A-Za-z0-9_./-]+', value) and not value.startswith('/') and '..' not in value.split('/'), 'Invalid navigation link')
    repo = plan['repo_id']
    require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo), 'Invalid repository identity')
    require(catalog['repo_id'] == repo, 'Catalog repository differs')
    expected = {}
    for model in plan['models']:
        study = studies[model['source_experiment']]
        require(model['model_key'] in MODELS and model['domain'] in DOMAINS and model['source_arm'] in methods_registry, 'Unknown model cell')
        prefix = f"experiments/{study['name']}/{MODELS[model['model_key']]}/{model['domain']}/{model['source_arm']}/seed-{model['seed']}/step-{model['terminal_step']:05d}"
        require(model['repo_prefix'] == prefix and prefix not in expected, 'Invalid or duplicate plan cell')
        expected[prefix] = model
    require(len(expected) == plan['expected_model_count'], 'Plan coverage differs')
    require(catalog['expected_model_count'] == len(expected) and catalog['verified_model_count'] == len(catalog['models']), 'Catalog count differs')
    rows = catalog['models']
    seen = set()
    for row in rows:
        prefix = row['repo_prefix']; require(prefix in expected and prefix not in seen, 'Unadmitted or duplicate catalog cell'); seen.add(prefix)
        m = expected[prefix]
        require((row['experiment'], row['model'], row['domain'], row['method'], row['seed'], row['step']) ==
                (studies[m['source_experiment']]['name'], MODELS[m['model_key']], m['domain'], m['source_arm'], m['seed'], m['terminal_step']), 'Catalog scientific identity differs')
        require(re.fullmatch(r'[0-9a-f]{40}', row['commit_sha']), 'Unpinned model revision')
    verified = {r['repo_prefix']: r for r in rows}
    total_selected = Counter(studies[m['source_experiment']]['name'] for m in plan['models'])
    total_verified = Counter(r['experiment'] for r in rows)
    complete = len(rows) == len(expected)
    coverage = plan.get('paper_coverage')
    record_only = {}
    if coverage is not None:
        record_only = coverage['record_only_by_source']
        require(set(record_only) <= set(studies), 'Record-only source is absent from study registry')
        require(all(type(n) is int and n >= 0 for n in record_only.values()), 'Invalid record-only count')
        require(coverage['deployable_model_exports'] == len(expected), 'Deployable-model coverage differs')
        require(coverage['record_only_count'] == sum(record_only.values()), 'Record-only census differs')
        require(coverage['logical_model_records'] == len(expected) + sum(record_only.values()), 'Scientific records and export census differ')
    root = ['---', 'library_name: transformers', 'tags:', '- reinforcement-learning', '- experimental-models', '---',
            '# ModeBench and Re:MaxRL · Research artifacts', '',
            'Terminal fine-tuned models for studying correctness and verified output support. Browse by study, base model, domain, method and seed; restore each export from its recorded immutable revision.', '',
            f"**{'Paper artifact collection' if coverage else 'Initial model collection'} · {len(rows)} / {len(expected)} selected exports verified. {'This selection is fully transferred.' if complete else 'Transfer is in progress.'}**", '',
            'The fixed archival selection contains admitted completed models from the source studies listed below. Counts describe archive availability, not live training progress or a replacement for the papers’ frozen analysis sets.', '',
            'The broader paper collection is being expanded to additional baselines, ablations and supporting data. This page reports only the listed verified exports; completion of this selection does not imply that every paper artifact has been archived.', '',
            ' · '.join(['[Models](#studies)'] + benchmark_nav + [f'[{label}]({sections[label]})' if label in sections else label + ' (in preparation)' for label in ('Data', 'Results', 'Reproducibility')]), '',
            '| Study | Question and model set | Verified / selected |', '| --- | --- | ---: |']
    for key, study in studies.items():
        exp = study['name']; root.append(f"| [{exp}]({landing(exp)}) | {study['title']} | {total_verified[exp]} / {total_selected[exp]} |")
    if coverage:
        missing_sources = ', '.join(f"{studies[key]['name']} ({count})" for key,count in sorted(record_only.items()) if count)
        root += ['', '## Artifact availability', '',
                 '| Scientific records selected | Deployable model exports | Verified model exports | Model weights unavailable |',
                 '| ---: | ---: | ---: | ---: |',
                 f"| {coverage['logical_model_records']} | {coverage['deployable_model_exports']} | {len(rows)} | {coverage['record_only_count']} |", '',
                 f"Model weights are unavailable for {coverage['record_only_count']} retained scientific records: {missing_sources}. Their metadata and admitted results remain part of the research record; they are excluded from the downloadable-model catalog.", '']
    root += ['', '[Study and comparator map](#studies) · [Restore a model](#restoring) · [Licenses](#licenses) · [Full catalog](catalog.json)', '', anchor('studies'), '## Find the models behind a result', '',
             '- **Retention across model scales:** [E78](experiments/E78/README.md#study-design), [E79](experiments/E79/README.md#study-design), [E80-R1](experiments/E80-R1/README.md#study-design).',
             '- **MaxRL × replay, Level 1:** [E118 and its original Dr.GRPO comparators](experiments/E118/README.md#comparators).',
             '- **Matched Level-2 factorial:** [E119](experiments/E119/README.md#study-design).',
             '- **Uniform versus fresh-frequency weighting:** [E120-R1 and its 45 original uniform comparators](experiments/E120-R1/README.md#comparators).', '',
             'The older comparator exports keep their E78/E79/E80-R1 names and appear once in the catalog. Training restarts do not create extra scientific cells. Model availability alone does not establish an admissible paired analysis; use the registered seed intersections and source checks in the papers.', '',
             anchor('restoring'), '## Restore one model', '',
             'Open a study table and select a verified model. Each link fixes the full upload commit, while `catalog.json` provides its complete `commit_sha` and `repo_prefix`. Download only that folder with `snapshot_download(repo_id=..., revision=commit_sha, allow_patterns=[repo_prefix + "/*"])` and verify the original file sizes and SHA-256 values in its `ARCHIVE_MANIFEST.json`.', '',
             'Exports include weights, configuration and tokenizer assets. They restore inference models; optimizer state and online replay banks are not included. Preserve the original domain prompt format, generation settings and verifier when reproducing a result.', '',
             anchor('licenses'), '## Base models and licenses', '',
             '| Base model | License and attribution |', '| --- | --- |',
             '| Qwen2.5-0.5B-Instruct | Apache License 2.0; original Alibaba attribution and modification notices retained. |',
             '| Falcon3-1B-Instruct | TII Falcon-LLM License 2.0 and incorporated acceptable-use policy. |',
             '| Qwen2.5-3B-Instruct | Qwen Research License Agreement, including its research/evaluation scope. |', '',
             '**Built with Qwen.** The Falcon3-1B experimental models in this archive are built using artificial intelligence technology from the Technology Innovation Institute. Each model folder carries its applicable `LICENSE`, `MODIFICATIONS.md`, and any required `Notice` or `ACCEPTABLE_USE_POLICY.html`.', '']
    if include_guides:
        root += ['[Complete restore example](RESTORE.md) · [Experiment reference](EXPERIMENTS.md) · [License texts and provenance](LICENSES.md)', '']
    result = {'README.md': '\n'.join(root)}
    for key, study in studies.items():
        exp = study['name']; selection = [m for m in plan['models'] if m['source_experiment'] == key]
        available = [r for r in rows if r['experiment'] == exp]
        methods = [method for method in methods_registry if any(m['source_arm'] == method for m in selection)]
        page = [f"# {exp} · {study['title']}", '', study['question'], '',
                '[Archive home](../../README.md) · [Study design](#study-design) · [Coverage](#archive-coverage) · [Models](#models) · [Comparators](#comparators)', '',
                anchor('study-design'), '## Study design', '', study['design'], '', study['paper'], '',
                'These are terminal exports from the registered eight-pass training horizon. The stored export-step label is 3073; the papers’ audited pass-8 evaluation endpoint is optimizer step 3072. The archive preserves that original export convention and does not imply an additional evaluated training update.', '',
                anchor('archive-coverage'), '## Archive coverage', '',
                '| Registered new training cells | Scientific records selected | Deployable exports selected | Verified exports |', '| ---: | ---: | ---: | ---: |',
                f"| {study['registered']} | {len(selection) + record_only.get(key, 0)} | {len(selection)} | {len(available)} |", '',
                'Scientific records preserve the admitted result identity. Deployable exports additionally have available model weights; verified exports have completed remote verification. Cells outside the fixed selection are not silently counted, and unverified model folders are not linked below.', '',
                '| Model | Verified / selected | Seeds in selection |', '| --- | ---: | --- |']
        if not selection:
            page = page[:-2]
        if record_only.get(key):
            page += ['', f"**Weights unavailable for {record_only[key]} scientific records in this study.** Their original metadata and admitted results are retained; no download link is provided for missing weights.", '']
        for model_key, label in MODELS.items():
            model_selection = [m for m in selection if m['model_key'] == model_key]
            if model_selection:
                page.append(f"| [{label}](#model-{model_key}) | {sum(r['model'] == label for r in available)} / {len(model_selection)} | {', '.join(map(str, sorted({m['seed'] for m in model_selection})))} |")
        page += ['', '## Methods and collections', '']
        for method in methods:
            page += [anchor('method-' + method), f'### {methods_registry[method][0]} · `{method}`', '', methods_registry[method][1], '']
            cells = [r for r in available if r['method'] == method]
            groups = sorted({(r['model'], r['domain']) for r in cells})
            if not groups:
                page += ['Transfer in progress; no verified collection links for this method yet.', '']
            for name, domain in groups:
                collection = f'https://huggingface.co/{repo}/tree/main/experiments/{exp}/{name}/{domain}/{method}'
                page.append(f'- [{name} · {DOMAINS[domain]}]({collection}) — {sum(r["model"] == name and r["domain"] == domain for r in cells)} verified exports. Exact immutable revisions are in the model tables below.')
            if groups: page.append('')
        page += [anchor('models'), '## Verified models', '',
                 'Browse by domain: ' + ' · '.join(f'[{label}](#domain-{domain})' for domain,label in DOMAINS.items()) + '.', '']
        for domain, label in DOMAINS.items():
            subset = [r for r in available if r['domain'] == domain]
            selected = sum(m['domain'] == domain for m in selection)
            page += [anchor('domain-' + domain), f'### {label}', '', f'**{len(subset)} / {selected}** selected exports verified.', '', _table(repo, subset, methods_registry), '']
        page += ['## Model-family shortcuts', '']
        for model_key, name in MODELS.items():
            if not any(m['model_key'] == model_key for m in selection): continue
            page += [anchor('model-' + model_key), f'### {name}', '']
            if any(r['model'] == name for r in available):
                page += [f'[Browse verified collections](https://huggingface.co/{repo}/tree/main/experiments/{exp}/{name}). Domain tables above provide exact model revisions.', '']
            else: page += ['Transfer in progress; this model family has no verified exports in the index yet.', '']
        page += [anchor('comparators'), '## Comparators and analysis scope', '']
        if key == 'e118':
            page += ['The MaxRL exports above supply two arms of the Level-1 factorial. Their original Dr.GRPO and Re:Dr.GRPO comparators are archived separately:', '',
                     '| Base model | Original Dr.GRPO pair |', '| --- | --- |',
                     '| Qwen2.5-0.5B-Instruct | [E78](../E78/README.md#models) |',
                     '| Falcon3-1B-Instruct | [E79](../E79/README.md#models) |',
                     '| Qwen2.5-3B-Instruct | [E80-R1](../E80-R1/README.md#models) |', '',
                     'Match the model, Level-1 domain and seed across all required arms. Do not infer a complete factorial block from a single archived export. Report partial seed intersections explicitly.', '']
        elif key == 'e120r1':
            comparators = [m for m in plan['models'] if m['campaign'] == 'e120_uniform_comparators']
            require(len(comparators) == 45 and all(m['source_arm'] == 'replay' for m in comparators), 'E120 comparator mapping differs')
            page += ['The 45 uniform comparators below are the original `replay` exports from E78/E79/E80-R1. They are cross-references, not additional archive entries. Match each to `fresh_frequency` by base model, domain and seed.', '',
                     '| Original source | Model | Domain | Selected seeds | Verified / selected |', '| --- | --- | --- | --- | ---: |']
            for source, model_key, domain in sorted({(m['source_experiment'],m['model_key'],m['domain']) for m in comparators}):
                group = [m for m in comparators if (m['source_experiment'],m['model_key'],m['domain']) == (source,model_key,domain)]
                source_name = studies[source]['name']
                page.append(f"| [{source_name}](../{source_name}/README.md#domain-{domain}) | {MODELS[model_key]} | {DOMAINS[domain]} | {', '.join(map(str, sorted(m['seed'] for m in group)))} | {sum(m['repo_prefix'] in verified for m in group)} / {len(group)} |")
            page += ['', 'The frozen primary analysis is Qwen2.5-0.5B across five domains and five seeds. Larger-model Graph/PantryPlan extensions are labeled separately. Extra modes are `distinct@8 − pass@8`; they are an unconditional count beyond the first correct mode, not diversity conditioned on equal correctness. Both arms share bank construction/update code, while policy-dependent bank contents may diverge.', '']
        elif key == 'e119':
            page += ['All four within-Level-2 arms are in this E119 study. For Level-1 context use [E78](../E78/README.md#models) and the Qwen2.5-0.5B subset of [E118](../E118/README.md#model-qwen05b). Keep the levels separate and use the common admitted seeds for each within-level contrast. Frozen-base admission and interim-training evaluations reported in the paper are separate artifacts; terminal model exports alone do not provide them.', '']
        elif key in ('e78', 'e79', 'e80r1'):
            page += ['Pair `replay` with `control` within the same base model, domain and seed. Selected replay exports also serve as uniform comparators for [E120-R1](../E120-R1/README.md#comparators); the [E118 factorial](../E118/README.md#comparators) reuses the original Dr.GRPO pairs. This archival link does not change the source experiment or analysis membership.', '']
        else:
            page += [study.get('comparators', 'Use the registered comparison set described above. This cohort is not automatically cross-listed into the original six-study comparisons.'), '']
        page += ['[Restore and verify a model](../../README.md#restoring) · [Applicable base-model licenses](../../README.md#licenses). The full `commit_sha`, source identity and file provenance are available through the root catalog and each pinned export manifest.', '']
        result[landing(exp)] = '\n'.join(page)
    for path, text in result.items():
        require(not any(s in text for s in ('/n/fs/', 'debug_job', 'job_id', 'run_stamp', 'token-file')), 'Private operational detail in public page')
    return result
