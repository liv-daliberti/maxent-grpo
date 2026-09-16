"""Register a rebuild of a domain whose original fit passed but was mis-centred.

`modebench_scale_domain_revision.register` exists for the ordinary case: an
original development fit FAILED, nothing was ever confirmed, and a new law is
registered before any held-out outcome can be observed. Two of its guards
therefore refuse Level 4 MathIR --

    require(previous['development_fit_pass'] is False, 'revise only a failed development fit')
    require(not (parent/level/'results/confirmation'/(domain+'.json')).exists(), ...)

-- because Level 4 MathIR's original development fit PASSED, and its held-out
receipt now exists. Both refusals are correct for what that function guarantees,
and neither is edited or bypassed here: `modebench_scale_domain_revision` is
pinned by sixteen sealed records including the Level 5 admission, and changing it
would invalidate the level that is admitted.

This registers the different, weaker thing honestly rather than dressing it up as
the stronger one. The information boundary it records is the true one:

    parent_development_fit_passed:  True   (the original was mis-centred, not failed)
    parent_holdout_observed:        True   (and it FAILED; the outcome is published)
    recalibrated_holdout_observed:  False

A domain registered here has already spent one held-out confirmation. That is a
real cost and the record says so, in `prior_heldout`, which pins the failed audit
and its metrics. Any dataset this produces carries a second confirmation, and
both belong in anything reported from it: a domain that needed two attempts is a
different claim from one that passed first time.

Because the outcome of the first attempt is known, the guard against selecting a
construction on test evidence has to be supplied by declaration instead of by
ignorance. `single_shot` is mandatory and written into the protocol: if the
recalibrated confirmation fails, the domain is finished and no further
recalibration may be registered against the same parent. `register` enforces that
by refusing when any sibling recalibration root for the same level and domain
already holds a confirmation receipt.

The protocol is written under the revision schema on purpose. Everything
downstream -- `materialize_pools`, `fit_domain`, `freeze_dataset`,
`confirm_domain`, `launch_inputs` -- then runs unmodified against it, so the
generation, freezing, grading and gate arithmetic are the same sealed code paths
that produced every admitted domain. Only the registration precondition differs,
and the extra fields record which precondition was used.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))

import materialize_modebench_scale as original
import fit_modebench_scale as common_fit
import modebench_scale_domain_revision as revision
from fit_modebench_level3 import atomic_new, file_sha

SCHEMA = revision.SCHEMA            # downstream authenticates against the revision schema
RECALIBRATION_KIND = 'modebench_scale_domain_recalibration_v1'
DEFAULT_PARENT = revision.DEFAULT_PARENT
SIBLING_ROOT = ROOT / 'var/data/modebench_scale_domain_revisions_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def prior_heldout(parent, level, domain):
    """The published audit that makes this a recalibration rather than a revision."""
    audit = Path(parent) / level / 'confirmation' / (domain + '.json')
    receipt = Path(parent) / level / 'results/confirmation' / (domain + '.json')
    require(receipt.is_file(), 'a recalibration requires an observed parent heldout receipt')
    require(audit.is_file(), 'a recalibration requires the published parent audit')
    value = read(audit)
    require(value['difficulty_matched'] is False,
            'the parent heldout passed; there is nothing to recalibrate')
    require(value['original_grader_replayed_attempts'] == 128 * 32,
            'parent audit did not replay the full heldout attempt set')
    return {'audit': str(audit), 'audit_sha256': file_sha(audit),
            'receipt': str(receipt), 'receipt_sha256': file_sha(receipt),
            'difficulty_matched': False, 'gates': value['gates'],
            'metrics': value['metrics'], 'differences': value['differences'],
            'target': value['target'],
            'bootstrap_delta_95': value.get('candidate_prompt_bootstrap_delta_95')}


def sibling_confirmations(level, domain):
    """Recalibration roots for this level/domain that already hold a receipt."""
    spent = []
    if not SIBLING_ROOT.is_dir():
        return spent
    for candidate in sorted(SIBLING_ROOT.iterdir()):
        protocol = candidate / 'protocol.json'
        if not protocol.is_file():
            continue
        value = read(protocol)
        if (value.get('recalibration_kind') == RECALIBRATION_KIND
                and value.get('level') == level and value.get('domain') == domain
                and (candidate / level / 'results/confirmation' / (domain + '.json')).is_file()):
            spent.append(str(candidate))
    return spent


def register(root, level, domain, *, parent=DEFAULT_PARENT, revision_number, candidate_module,
             single_shot=True, rationale):
    """Register a recalibration root; downstream tooling then treats it as a revision."""
    candidates = revision.candidate_provider(candidate_module)
    root, parent = Path(root).resolve(), Path(parent).resolve()
    require(not root.exists(), 'fresh recalibration root required')
    require(single_shot is True, 'a recalibration must be registered single-shot')
    require(isinstance(rationale, str) and len(rationale) >= 80,
            'a recalibration must record why the original was mis-centred')
    require(domain in candidates.PROFILES and len(candidates.PROFILES[domain]) == 4,
            'four registered candidate laws required')

    p = original.authenticate(parent / 'protocol.json')
    recipe_path = parent / level / 'recipes' / (domain + '.json')
    previous = read(recipe_path)
    require(previous == common_fit.fit_domain(parent, level, domain, publish=False),
            'parent fit is not reproducible')
    # The mirror image of revision.register: this path exists precisely for the
    # domain whose development fit passed while sitting off centre.
    require(previous['development_fit_pass'] is True,
            'a failed development fit belongs in modebench_scale_domain_revision.register')
    prior = prior_heldout(parent, level, domain)
    spent = sibling_confirmations(level, domain)
    require(not spent, 'single-shot rule: a recalibration for this domain already confirmed: '
            + ', '.join(spent))

    pin_paths = set(candidates.source_paths()) | {
        Path(__file__), Path(revision.__file__), Path(original.__file__), Path(common_fit.__file__),
        ROOT / 'ops/exp_scaling/modebench_scale_source_disjointness.py',
        parent / 'protocol.json', recipe_path,
        Path(prior['audit']), Path(prior['receipt'])}
    pins = {**p['files_sha256'], **{str(Path(f).resolve()): file_sha(Path(f)) for f in pin_paths}}
    pins.update(previous['input_sha256'])

    value = {
        'schema': SCHEMA, 'recalibration_kind': RECALIBRATION_KIND,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'root': str(root), 'parent_root': str(parent),
        'level': level, 'domain': domain, 'revision': revision_number,
        'parent_protocol_sha256': file_sha(parent / 'protocol.json'),
        'parent_recipe_sha256': file_sha(recipe_path),
        'models': p['models'], 'targets': {domain: p['targets'][domain]},
        'histograms': {domain: p['histograms'][domain]},
        'split_sizes': p['split_sizes'], 'tolerances': p['tolerances'],
        'selection_seed': original.SELECTION_SEED,
        'candidate_module': candidate_module,
        'candidate_profiles': candidates.PROFILES[domain],
        'sampling': p['sampling'], 'fit': p['fit'],
        'draw_labels': {phase: revision.labels(level, domain, revision_number, phase)
                        for phase in ('dev', 'eval')},
        'generation_seed_rule': 'sha256([schema, level, domain, revision, split, tier])[:12]',
        'exclusion_rule': ('Snapshot all historical and currently materialized ModeBench rows '
                           'before each generation; verify cross-source disjointness before '
                           'confirmation and release.'),
        'rationale': rationale,
        'prior_heldout': prior,
        'single_shot': True,
        'single_shot_rule': ('This domain has already spent one held-out confirmation. If the '
                             'recalibrated confirmation fails, the domain is finished: no further '
                             'recalibration may be registered against this parent, and both '
                             'outcomes are reported.'),
        'information_boundary': {
            'parent_development_fit_passed': True,
            'parent_holdout_observed': True,
            'parent_holdout_difficulty_matched': False,
            'recalibrated_holdout_observed': False,
            'source_choice': 'development_only',
            'caveat': ('Unlike a revision, the parent held-out outcome was known when this law '
                       'was chosen. The guard against selecting on test evidence is the declared '
                       'single-shot rule, not ignorance of the outcome.'),
        },
        'files_sha256': pins,
    }
    root.mkdir(parents=True)
    atomic_new(root / 'protocol.json', value)
    atomic_new(root / 'protocol.sha256.json', {'sha256': file_sha(root / 'protocol.json')})
    revision.authenticate(root)     # must satisfy the sealed downstream contract
    return value
