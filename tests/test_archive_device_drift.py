"""``st_dev`` is a mount identifier, not a file identifier.

The archive deletes a local weight file only after re-verifying it against the
manifest recorded at upload. That check compared the whole stat identity, which
includes ``st_dev`` -- a number NFS assigns per host and per mount. The same
untouched export therefore failed verification from any machine other than the
one that wrote it, which is what left the archive unretirable.

These tests pin the exception to exactly one field. Everything that identifies
content -- the digest, the size, the inode, mtime and ctime -- must still match,
so a file that was replaced, rewritten, truncated or swapped for another inode
continues to fail.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops'))
from model_archive_verification import _device_only_difference  # noqa: E402


def record(**overrides):
    base = {
        'local_path': '/export/model.safetensors',
        'relative_path': 'model.safetensors',
        'path_in_repo': 'experiments/E1/m/model.safetensors',
        'size': 4096,
        'sha256': 'a' * 64,
        'git_blob_sha1': 'b' * 40,
        'stat': {'dev': 48, 'ino': 12345, 'mtime_ns': 1, 'ctime_ns': 2},
    }
    base.update(overrides)
    return base


def with_stat(base, **changes):
    other = copy.deepcopy(base)
    other['stat'].update(changes)
    return other


def test_device_only_change_is_recognised():
    before = record()
    after = with_stat(before, dev=47)
    assert _device_only_difference(after, before) == {'recorded_device': 48, 'observed_device': 47}


def test_identical_records_are_not_drift():
    before = record()
    assert _device_only_difference(copy.deepcopy(before), before) is None


@pytest.mark.parametrize('field,value', [
    ('sha256', 'c' * 64),
    ('git_blob_sha1', 'd' * 40),
    ('size', 4097),
    ('local_path', '/elsewhere/model.safetensors'),
    ('path_in_repo', 'experiments/E1/m/other.safetensors'),
])
def test_any_content_field_change_still_fails(field, value):
    before = record()
    after = with_stat(before, dev=47)
    after[field] = value
    assert _device_only_difference(after, before) is None, (
        f'{field} differing must never be excused as device drift')


@pytest.mark.parametrize('field,value', [('ino', 999), ('mtime_ns', 9), ('ctime_ns', 9)])
def test_other_stat_fields_still_fail(field, value):
    before = record()
    after = with_stat(before, dev=47, **{field: value})
    assert _device_only_difference(after, before) is None, (
        f'stat.{field} differing must never be excused as device drift')


def test_a_replaced_file_with_the_same_device_still_fails():
    before = record()
    after = copy.deepcopy(before)
    after['sha256'] = 'e' * 64
    assert _device_only_difference(after, before) is None


def test_missing_or_extra_keys_fail_closed():
    before = record()
    after = with_stat(before, dev=47)
    after.pop('git_blob_sha1')
    assert _device_only_difference(after, before) is None
    extra = with_stat(before, dev=47)
    extra['unexpected'] = 1
    assert _device_only_difference(extra, before) is None


def test_malformed_stat_fails_closed():
    before = record()
    after = with_stat(before, dev=47)
    after['stat'] = None
    assert _device_only_difference(after, before) is None
    missing = with_stat(before, dev=47)
    del missing['stat']['ino']
    assert _device_only_difference(missing, before) is None


# --- the second pre-deletion check, inside the storage lock -------------------

from model_archive_verification import identity_matches  # noqa: E402


def stat_identity(**overrides):
    base = {'dev': 48, 'ino': 12345, 'mtime_ns': 1, 'ctime_ns': 2}
    base.update(overrides)
    return base


def test_identity_matches_allows_device_only():
    assert identity_matches(stat_identity(dev=47), stat_identity()) is True


def test_identity_matches_accepts_exact_equality():
    assert identity_matches(stat_identity(), stat_identity()) is True


@pytest.mark.parametrize('field', ['ino', 'mtime_ns', 'ctime_ns'])
def test_identity_matches_rejects_other_fields(field):
    observed = stat_identity(dev=47, **{field: 999})
    assert identity_matches(observed, stat_identity()) is False, (
        f'{field} differing must not be excused')


def test_identity_matches_rejects_shape_mismatch():
    observed = stat_identity(dev=47)
    del observed['ctime_ns']
    assert identity_matches(observed, stat_identity()) is False
    extra = stat_identity(dev=47)
    extra['extra'] = 1
    assert identity_matches(extra, stat_identity()) is False
