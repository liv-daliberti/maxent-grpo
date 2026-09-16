"""Public paper-to-archive guide; source-family descriptions, no artifact admission."""
import re

INITIAL_STUDIES = ('E78', 'E79', 'E80-R1', 'E118', 'E119', 'E120-R1')
FAMILIES = {
    'grpo': {'label': 'Plain GRPO', 'sources': ('E95', 'E114'),
             'description': 'Reward-standardized group-relative learning. E95 supplies the cross-scale cohort; E114 extends the Qwen2.5-3B seed set.'},
    'ucpo': {'label': 'UCPO', 'sources': ('E97', 'E99', 'E115'),
             'description': 'An on-policy comparator that reallocates positive advantage inside the current rollout group. E97/E115 supply Qwen cohorts; E99 supplies Falcon.'},
    'sparse-rlep': {'label': 'Sparse RLEP-Dr', 'sources': ('E98-R1', 'E100', 'E116'),
                    'description': 'Verified-success replay using observed frequency, without canonical-key balancing. E98-R1/E116 supply Qwen cohorts; E100 supplies Falcon. Repaired source identities remain explicit.'},
    'fixed-semantic': {'label': 'Fixed Semantic-MaxEnt, with and without replay', 'sources': ('E81', 'E82', 'E83', 'E85', 'E86'),
                       'description': 'Detached, smoothed surprisal over verified canonical outcomes is added to Dr.GRPO, with a fixed coefficient. The replay and no-replay arms remain separate; E85 contains the registered PantryPlan semantic repair.'},
    'verified-support': {'label': 'Verified-support semantic + proposal comparison', 'sources': ('E112-R1', 'E109'),
                         'description': 'A bundled comparison adds semantic regularization and isolated proposal sampling to Re:Dr. E112-R1 is the treatment family; E109 supplies repaired Python comparators where required. The bundle does not isolate semantic regularization from proposal sampling.'},
    'historical-semantic': {'label': 'Historical Qwen2.5-3B fixed-semantic probe', 'sources': ('E87',),
                            'description': 'A one-seed (70), five-domain fixed-semantic-plus-replay probe against the original E80-R1 references. It is historical supporting material, separate from current multi-seed comparison estimates.'},
    'fixed-bank': {'label': 'Fixed-bank survival mechanism', 'sources': ('E121',),
                   'description': 'Five Graph Coloring runs measure likelihood and survival of retained exemplars after the bank is frozen. The model weights alone do not reproduce the fixed-bank identities or telemetry.'},
}


def render_experiment_guide(available_studies=INITIAL_STUDIES, *, available_sections=None):
    """Link only study/data indexes explicitly confirmed available by the publisher."""
    available = set(available_studies)
    if any(not re.fullmatch(r'E[0-9]+(?:-[A-Z0-9]+)*', name) for name in available):
        raise ValueError('Invalid public experiment name')
    sections = available_sections or {}
    if set(sections) - {'Data', 'Results', 'Reproducibility'}:
        raise ValueError('Unknown guide section')
    for value in sections.values():
        if not re.fullmatch(r'[A-Za-z0-9_./-]+', value) or value.startswith('/') or '..' in value.split('/'):
            raise ValueError('Invalid public section path')
    def link(name, anchor='models'):
        return f'[{name}](experiments/{name}/README.md#{anchor})' if name in available else f'{name} — index in preparation'
    def family(name):
        item = FAMILIES[name]
        return f"| {item['label']} | {', '.join(link(source) for source in item['sources'])} | {item['description']} |"
    lines = ['# Experiments · Paper-to-archive guide', '',
             '[Archive home](README.md) · [Direct comparators](#direct-comparators) · [Semantic ablations](#semantic-ablations) · [Data and reproducibility](#reproducibility)', '',
             'This guide connects the paper’s comparisons to their original experiment families. It describes scientific identity and artifact scope; it does not turn an unfinished transfer into an available model.', '',
             'The initial fixed selection contains 418 completed terminal exports from six experiments. Additional paper baselines, semantic ablations, mechanism studies and supporting data are being admitted and transferred separately. Current verified counts and immutable model revisions are in the root catalog and study indexes. New source families are linked only after their indexes are published.', '',
             '<a id="core-replay"></a>', '## Core replay comparisons', '',
             '| Paper comparison | Original model families | How to match the exports |', '| --- | --- | --- |',
             f"| Experiment 1: retention across scales | {link('E78')}, {link('E79')}, {link('E80-R1')} | Match `replay` and `control` within base model, domain and seed. Preserve source-integrity exclusions and the paper’s exact paired intersection. |",
             f"| Experiment 2: MaxRL × replay, Level 1 | {link('E118', 'comparators')} plus the original E78/E79/E80-R1 Dr.GRPO pairs | E118 supplies `maxrl` and `replay_maxrl`. The older families supply `control` and `replay`; use the common four-arm intersection for the interaction. |",
             f"| Experiment 3: matched Level 2 | {link('E119', 'study-design')}; Level 1 context from {link('E78')} and {link('E118')} | All four E119 methods train on Level 2. Keep its prompt set and level labels distinct from the Level 1 Qwen2.5-0.5B references. |", '',
             '<a id="frequency-ablation"></a>', '## Uniform versus fresh-frequency replay', '',
             f"{link('E120-R1', 'study-design')} contains the fresh-frequency treatment. Its 45 uniform comparators remain original `replay` exports: 25 from E78 (five Qwen2.5-0.5B domains), 10 from E79 (Falcon Graph/PantryPlan), and 10 from E80-R1 (Qwen2.5-3B Graph/PantryPlan). They are cross-references, never duplicate model entries.", '',
             f"{link('E120-R1', 'comparators')} lists the exact source/model/domain/seed mapping. The frozen primary analysis uses the 25 Qwen2.5-0.5B pairs; larger-model extensions remain separate. The direction is uniform minus fresh-frequency. Extra modes are `distinct@8 − pass@8`, an unconditional count beyond the first correct mode rather than diversity conditioned on equal correctness.", '',
             '<a id="direct-comparators"></a>', '## Direct comparators', '',
             'These source families support the paper’s comparison of estimator normalization, on-policy diversity incentives, and success memory. Archive availability is determined by verified model receipts; the numbers in a frozen paper analysis are not transfer counts.', '',
             '| Comparator | Original source families | Active distinction and scope |', '| --- | --- | --- |',
             family('grpo'), family('ucpo'), family('sparse-rlep'), '',
             f"The Dr.GRPO and Re:Dr reference families are {link('E78')}, {link('E79')} and {link('E80-R1')}. Binary MaxRL alternatives are in {link('E118')}. Fixed semantic-entropy alternatives are described below. Apply the paper’s model/domain/seed restrictions rather than pooling all stored exports into a new analysis.", '',
             '<a id="semantic-ablations"></a>', '## Semantic ablations and supporting studies', '',
             'Semantic estimators are comparators to canonical replay, not interchangeable names for Re:Dr. Their treatment definitions and source repairs remain explicit.', '',
             '| Family | Original source studies | Treatment and interpretation |', '| --- | --- | --- |',
             family('fixed-semantic'), family('verified-support'), family('historical-semantic'), '',
             'The fixed-semantic predictor is detached and smoothed; the implemented update is not claimed to be an unbiased entropy gradient. The verified-support comparison bundles semantic regularization with proposal sampling. Adaptive controllers, group-centering diagnostics and open-bank development branches are excluded from that reported E112-R1 comparison and must not be relabeled as its treatment.', '',
             'E87 has one paired seed per domain. Its archive entry preserves that historical scope; it is not a multi-seed estimate and contributes no bootstrap uncertainty or pooled-domain effect to the current comparison.', '',
             '<a id="mechanism-studies"></a>', '## Mechanism studies', '',
             '| Study | Source | Required artifacts |', '| --- | --- | --- |', family('fixed-bank'), '',
             'For fixed-bank analyses retain the bank identity, teacher-forced likelihood traces, scoring conventions and generated result tables. A terminal Transformers export does not restore the online replay bank or optimizer trajectory.', '',
             '<a id="reproducibility"></a>', '## Data, results and reproducibility', '',
             ' · '.join(f'[{name}]({sections[name]})' if name in sections else name + ' — in preparation' for name in ('Data', 'Results', 'Reproducibility')), '',
             'The supporting archive includes admitted benchmark manifests and splits, original analysis inputs, numerical result tables, figure source data, protocols and source-identity records as they are verified and published. Manuscript snapshots retain their own dates and cohort membership. Reproduce a stated figure from its recorded inputs rather than substituting a newer endpoint or larger seed set.', '',
             'For the initial six model families, the original terminal folder step 3073 is a bookkeeping export of the unchanged policy evaluated at optimizer step 3072. It is not a 3073rd optimizer update. Frozen-base admission, pass-by-pass evaluations and interim checkpoints are separate artifacts; final model weights alone do not recreate them.', '',
             'Use the full 40-character model commit together with its repository subfolder. Check file sizes and SHA-256 digests against the pinned export manifest. Base-model licenses and required notices accompany each model; public availability does not replace their terms.', '']
    return '\n'.join(lines)
