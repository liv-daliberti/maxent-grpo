import audit_constructive_code_sources as source_audit

from audit_constructive_code_sources import (
    build_candidate_index,
    build_audit,
    checker_is_reference_independent,
    normalize_text,
    plus_overlay_name,
    python_submission_count,
    select_audited_candidate_pairs,
)


def _plus_row(**overrides):
    row = {
        "source": "Codeforces",
        "id": "100_A",
        "title": "A Witness",
        "description": "Print any valid witness.\r\n",
        "checker": "int main(){ inf.readInt(); ouf.readInt(); }",
        "correct_submissions": [
            {"language": "py3"},
            {"language": "python3"},
        ],
        "true_positive_rate": 0.9,
        "true_negative_rate": 0.95,
        "_source_shard": "ccplus_5x/part-00000.parquet",
    }
    row.update(overrides)
    return row


def _overlay_row(**overrides):
    row = {
        "name": "100_A. A Witness",
        "description": "Print any valid witness.\n",
        "checker": "int main(){ inf.readInt(); ouf.readInt(); }",
        "_source_shard": "data/train-00000-of-00001.parquet",
    }
    row.update(overrides)
    return row


def test_normalize_text_handles_unicode_newlines_and_spacing():
    assert normalize_text("e\u0301\r\n  value") == "\u00e9 value"


def test_reference_independent_checker_rule_is_conservative():
    assert checker_is_reference_independent(
        'setName("ans is irrelevant"); inf.readInt(); ouf.readInt();'
    )
    assert not checker_is_reference_independent(
        "inf.readInt(); ouf.readInt(); ans.readInt();"
    )
    assert not checker_is_reference_independent("ouf.readInt();")
    assert not checker_is_reference_independent("")


def test_python_submission_count_and_overlay_name():
    assert (
        python_submission_count(
            [
                {"language": "Py3"},
                {"language": "cpp"},
                {"language": "pypy3"},
                None,
            ]
        )
        == 2
    )
    assert plus_overlay_name(_plus_row()) == "100_A. A Witness"
    assert (
        plus_overlay_name(_plus_row(source="AIZU", id="p00001")) == "p00001. A Witness"
    )


def test_build_audit_counts_funnel_and_first_rejections():
    plus_rows = [
        _plus_row(),
        _plus_row(id="101_A", title="Low TPR", true_positive_rate=0.89),
        _plus_row(
            id="102_A",
            title="Reference Checker",
            checker="inf.readInt(); ouf.readInt(); ans.readInt();",
        ),
        _plus_row(
            id="103_A",
            title="No Python",
            correct_submissions=[{"language": "cpp"}],
        ),
        _plus_row(id="104_A", title="Missing Overlay"),
        _plus_row(id="105_A", title="Statement Drift"),
        _plus_row(id="106_A", title="Checker Drift"),
    ]
    overlay_rows = [
        _overlay_row(),
        _overlay_row(
            name="105_A. Statement Drift",
            description="Different statement",
        ),
        _overlay_row(
            name="106_A. Checker Drift",
            description="Print any valid witness.",
            checker="int main(){ inf.readInt(); ouf.readLong(); }",
        ),
    ]

    audit = build_audit(
        plus_rows,
        overlay_rows,
        {"repo": "plus"},
        {"repo": "overlay"},
        verified_threshold=0.9,
        minimum_python_submissions=2,
    )

    assert audit["audited_candidate_count"] == 1
    assert audit["eligibility_funnel"] == {
        "all_plus_5x_rows": 7,
        "verified_tpr_tnr": 6,
        "reference_independent_checker_lower_bound": 5,
        "python_supported": 4,
        "unique_cross_source_name_match": 3,
        "normalized_statement_hash_match": 2,
        "normalized_checker_hash_match": 1,
    }
    assert audit["first_rejection_counts"] == {
        "checker_requires_reference_or_is_ambiguous": 1,
        "insufficient_correct_python_submissions": 1,
        "missing_overlay_name": 1,
        "normalized_checker_hash_mismatch": 1,
        "normalized_statement_hash_mismatch": 1,
        "not_codecontests_plus_verified": 1,
    }
    assert audit["row_integrity"]["codecontests_o_split_counts"] == {"train": 3}
    assert audit["row_integrity"]["codecontests_o_empty_names"] == 0
    assert audit["access_contract"]["full_dataset_materialized"] is False


def test_build_audit_ignores_empty_overlay_name_for_raw_alignment():
    audit = build_audit(
        [_plus_row(source="AIZU", id="", title="")],
        [_overlay_row(name="", description="", checker="")],
        {"repo": "plus"},
        {"repo": "overlay"},
        verified_threshold=0.9,
        minimum_python_submissions=2,
    )

    assert audit["raw_cross_source_alignment"] == {
        "unique_exact_name_matches": 0,
        "normalized_statement_hash_matches": 0,
        "normalized_checker_hash_matches": 0,
    }
    assert audit["row_integrity"]["codecontests_o_empty_names"] == 1


def test_candidate_index_is_exactly_bound_to_count_only_audit():
    plus_rows = [_plus_row(_source_row_index=7)]
    overlay_rows = [_overlay_row(_source_row_index=3)]
    audit = build_audit(
        plus_rows,
        overlay_rows,
        {"repo": "plus"},
        {"repo": "overlay"},
        verified_threshold=0.9,
        minimum_python_submissions=2,
    )
    pairs = select_audited_candidate_pairs(
        plus_rows,
        overlay_rows,
        verified_threshold=0.9,
        minimum_python_submissions=2,
    )

    index = build_candidate_index(pairs, audit)

    assert index["record_count"] == audit["audited_candidate_count"] == 1
    assert (
        index["source_audit"]["candidate_keys_sha256"]
        == audit["audited_candidate_keys_sha256"]
    )
    assert index["records"][0]["problem_key"] == "Codeforces:100_A:A Witness"
    assert index["records"][0]["codecontests_plus"]["source_row_index"] == 7
    assert index["records"][0]["codecontests_o"]["source_row_index"] == 3
    assert index["records"][0]["manual_review"]["status"] == "pending"


def test_remote_reader_passes_known_size_to_file_open(monkeypatch):
    captured = {}

    class FakeHandle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeFilesystem:
        def open(self, url, mode, **kwargs):
            captured.update(url=url, mode=mode, kwargs=kwargs)
            return FakeHandle()

    class FakeTable:
        def to_pylist(self):
            return [{"id": "100_A"}]

    class FakeParquetFile:
        def __init__(self, _handle):
            pass

        def read(self, **_kwargs):
            return FakeTable()

    monkeypatch.setattr(
        source_audit.fsspec,
        "filesystem",
        lambda protocol: FakeFilesystem(),
    )
    monkeypatch.setattr(source_audit.pq, "ParquetFile", FakeParquetFile)

    rows = source_audit._read_parquet_shard(
        "owner/repo",
        "a" * 40,
        "data/file.parquet",
        12345,
        ("id",),
        attempts=1,
    )

    assert rows == [
        {
            "id": "100_A",
            "_source_shard": "data/file.parquet",
            "_source_row_index": 0,
        }
    ]
    assert captured["mode"] == "rb"
    assert captured["kwargs"]["size"] == 12345
    assert captured["kwargs"]["block_size"] == 64 * 1024
