"""Reviewed labels for additional paper cohorts; no admission or model operations.

Source identifiers remain original experiment names. This module only supplies
public descriptions to the metadata renderer after a combined plan is approved.
"""
ADDITIONAL_METHODS = {
    'grpo_plain_control': ('GRPO', 'Reward-standardized group-relative task update, with no auxiliary replay derivative.'),
    'ucpo': ('UCPO', 'Reallocates positive advantage within the current on-policy rollout group.'),
    'rlep_dr_sparse': ('Sparse RLEP-Dr', 'Sparse replay of verified successes in observed frequency, without canonical-key balancing.'),
    'semantic': ('Fixed Semantic-MaxEnt + Re:Dr.GRPO', 'Fixed detached surprisal over verified canonical outcomes, added to Dr.GRPO with verified replay.'),
    'semantic_only': ('Fixed Semantic-MaxEnt', 'The fixed semantic advantage with the replay derivative disabled.'),
    'verified_support_discovery': ('Verified-support semantic + proposal bundle', 'Re:Dr.GRPO augmented with semantic regularization and an isolated verified proposal sampler; a bundled comparison.'),
    'fixed_bank_survival': ('Fixed-bank survival', 'Re:Dr.GRPO with a bank frozen at the registered boundary and exemplar-likelihood survival telemetry.'),
}


def study(name, title, registered, design, scope, comparators, question='Which aspect of the update changes correctness and retained verified support?'):
    return {'name': name, 'title': title, 'registered': registered, 'question': question,
            'design': design, 'paper': scope, 'comparators': comparators}


DIRECT = 'Direct comparator family. Apply the paper’s frozen model/domain/seed inclusion rules; scientific record counts and verified downloadable model counts are separate.'
FIXED = 'Fixed-semantic ablation family. This comparator is separate from canonical replay alone and from the verified-support proposal bundle.'
SUPPORT = 'Reported exploratory bundled comparison. Semantic regularization and proposal sampling are introduced together; their individual effects are not isolated.'
ADDITIONAL_STUDIES = {
    'e95': study('E95', 'Plain GRPO · initial cross-scale cohort', 55,
                 'Qwen2.5-0.5B and Falcon3-1B: five domains and five seeds each; Qwen2.5-3B: five domains at seed 70.', DIRECT,
                 'Use the original E78/E79/E80-R1 Dr.GRPO references. Missing weights remain scientific records only and are not advertised as downloadable models.'),
    'e114': study('E114', 'Plain GRPO · Qwen2.5-3B seed extension', 20,
                  'Qwen2.5-3B across five domains and seeds 71–74; together with E95 seed 70 this forms the five-seed plain-GRPO comparison.', DIRECT,
                  'Pair with E80-R1 Dr.GRPO by domain and seed. Keep the E114 source identity for extension exports.'),
    'e97': study('E97', 'UCPO · Qwen2.5-0.5B initial domains', 15,
                 'Qwen2.5-0.5B across Graph Coloring, Python Factors and PantryPlan, with five seeds per domain.', DIRECT,
                 'Use E78 matched Dr.GRPO/Re:Dr.GRPO references. E115 supplies the remaining Qwen2.5-0.5B domains.'),
    'e99': study('E99', 'UCPO · Falcon3-1B', 25,
                 'Falcon3-1B across five domains and five registered seeds.', DIRECT,
                 'Use original E79 comparators and the common admitted seed intersection.'),
    'e115': study('E115', 'UCPO · domain and scale extension', 35,
                  'Ten Qwen2.5-0.5B cells extend to Countdown and MathIR; 25 Qwen2.5-3B cells are registered separately.', DIRECT,
                  'The current paper’s admitted Qwen2.5-0.5B extension is distinct from the separately registered larger-model extension. Stored availability does not add a cell to a frozen analysis.'),
    'e98r1': study('E98-R1', 'Sparse RLEP-Dr · Qwen2.5-0.5B', 15,
                   'Corrected sparse-success-replay recipe for Graph Coloring, Python Factors and PantryPlan, five seeds each.', DIRECT,
                   'Use E78 references; E116 supplies further domains. Preserve the corrected E98-R1 source identity rather than relabeling it as superseded E98.'),
    'e100': study('E100', 'Sparse RLEP-Dr · Falcon3-1B', 25,
                  'Falcon3-1B across five domains and five registered seeds, with registered execution repairs retained in source provenance.', DIRECT,
                  'Use E79 references and the exact admitted cohort; missing or excluded seeds are not imputed.'),
    'e116': study('E116', 'Sparse RLEP-Dr · domain and scale extension', 35,
                  'Ten Qwen2.5-0.5B cells extend to Countdown and MathIR; 25 Qwen2.5-3B cells are registered separately.', DIRECT,
                  'Use matched source-model/domain/seed references. The replay collection and completion stages belong to the same scientific treatment cell.'),
    'e81': study('E81', 'Fixed semantic + replay · Qwen2.5-0.5B', 25,
                 'Fixed semantic advantage added to verified replay across five domains and five seeds.', FIXED,
                 'Compare with E78 Re:Dr.GRPO. Use E85 repaired PantryPlan exports where specified by the paper; retain E85 as their source experiment.'),
    'e82': study('E82', 'Fixed semantic + replay · Falcon3-1B', 25,
                 'Fixed semantic advantage added to Falcon3-1B verified replay across five domains and five seeds.', FIXED,
                 'Compare with E79 Re:Dr.GRPO. The source audit and registered repair history determine the admissible exports.'),
    'e83': study('E83', 'Fixed semantic without replay · Qwen2.5-0.5B', 25,
                 'Fixed semantic advantage with zero replay derivative across five domains and five seeds.', FIXED,
                 'Compare with E78 Dr.GRPO and the semantic-plus-replay arm. The paper uses E85 repaired PantryPlan exports for this family.'),
    'e85': study('E85', 'PantryPlan semantic-interface repairs', 15,
                 'Registered PantryPlan replacements preserve their parent scientific treatment: E81 replay+semantic, E83 semantic-only, and the Falcon E82 repair family.', FIXED,
                 'The two Qwen methods keep distinct semantic/semantic_only archive keys. Original ledger labels and parent identities are retained separately so repaired no-replay models cannot collide with replay models.'),
    'e86': study('E86', 'Fixed semantic without replay · Falcon3-1B', 25,
                 'Falcon3-1B fixed semantic advantage with zero replay derivative, with the semantic interface already repaired.', FIXED,
                 'Compare with E79 Dr.GRPO and the admissible semantic-plus-replay family.'),
    'e87': study('E87', 'Historical fixed-semantic probe · Qwen2.5-3B', 5,
                 'One registered seed 70 across five domains; fixed semantic advantage is added to the E80-R1-style replay recipe.',
                 'Historical descriptive supporting study, explicitly separated from multi-seed current-paper comparisons. A single seed supports no seed-bootstrap interval or pooled-domain causal claim.',
                 'Use E80-R1 seed 70 control/replay references only; do not merge this one-seed probe into a multi-seed fixed-semantic estimate.'),
    'e109': study('E109', 'Repaired Python replay comparators', 15,
                  'Five Python Factors seeds at each of three registered model scales, preserving the repaired prompt/action surface.',
                  'Comparator family for the verified-support analysis; only the paper’s admitted scales and source pairings belong to the reported contrast.',
                  'Pair the appropriate repaired Re:Dr.GRPO source with E112-R1 by model and seed. These are distinct source exports, not replacements for all E78/E79/E80-R1 uses.'),
    'e112r1': study('E112-R1', 'Verified-support semantic + proposal bundle', 75,
                    'Three registered model scales × five domains × five seeds. The reported two-scale contrast uses its exact common admitted intersection.', SUPPORT,
                    'Use original replay comparators, with E109 repaired Python sources where required. Adaptive controllers and open-bank development branches are excluded from this reported treatment, not from every historical artifact in the broader project.'),
    'e121': study('E121', 'Fixed-bank exemplar survival', 5,
                  'Five Qwen2.5-0.5B Graph Coloring runs measure retained exemplar likelihood after a registered bank-freeze boundary.',
                  'Mechanism study. Teacher-forced likelihood and bank identity telemetry are separate from end-model correctness and support measurements.',
                  'There is no matched no-replay fixed-bank arm in this study; its trajectories alone do not identify a causal replay effect.',
                  question='How do stored exemplar likelihoods evolve after the replay bank is frozen?'),
}


def registry_for(plan):
    """Definitions for admitted plan sources plus explicitly listed record-only sources."""
    sources = {m['source_experiment'] for m in plan['models']}
    sources.update(plan.get('paper_coverage', {}).get('record_only_by_source', {}))
    selected = {key: value for key, value in ADDITIONAL_STUDIES.items() if key in sources}
    return selected, dict(ADDITIONAL_METHODS)
