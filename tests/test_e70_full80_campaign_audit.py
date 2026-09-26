import audit_e70_full80_campaign as audit


def _stage_a(status="pass", terminal=40):
    return {
        "status": status,
        "summary": {"identity_runs": 40, "terminal_runs": terminal},
    }


def _stage_b(status="pass", count=10):
    return ({"jobs": {str(index): index for index in range(count)}}, {"status": status})


def test_full80_pass_requires_all_four_stage_b_rows_and_stage_a():
    rows = {name: _stage_b() for name in audit.STAGE_B}
    payload = audit.summarize(
        _stage_a(), rows, {name: {"status": "pass"} for name in audit.CONSTRUCTIVE_QUALIFICATIONS}
    )
    assert payload["status"] == "pass"
    assert payload["summary"]["submitted_cells"] == 80
    assert payload["summary"]["audited_terminal_cells"] == 80
    coverage = payload["semantic_domain_coverage"]
    assert coverage["status"] == "incomplete"
    assert coverage["missing_requested_domains"] == ["constructive_code"]
    assert coverage["independent_domains_covered"] == 7
    assert coverage["replacement_is_independent_domain"] is False



def test_missing_geometry_shift_replacement_remains_in_progress_not_complete():
    rows = {name: _stage_b() for name in audit.STAGE_B}
    rows["point_maze_geometry_shift"] = ({}, {})
    payload = audit.summarize(_stage_a(), rows, {"v6_gate": {"status": "pass"}})
    assert payload["status"] == "in_progress"
    assert payload["summary"]["submitted_cells"] == 70
    assert payload["summary"]["audited_terminal_cells"] == 70


def test_failed_intermediate_qualification_is_recorded_without_integrity_failure():
    rows = {name: _stage_b() for name in audit.STAGE_B}
    payload = audit.summarize(
        _stage_a(status="in_progress", terminal=25),
        rows,
        {"v6_viability": {"status": "fail"}},
    )
    assert payload["status"] == "in_progress"
    assert payload["negative_qualification_outcomes"] == ["v6_viability"]
    assert payload["violations"] == []


def test_cell_registry_keeps_exact_job_and_progress_state():
    stage_a = {
        "domains": {
            "mathir": {
                "runs": [
                    {
                        "arm": "grpo",
                        "seed": 47,
                        "job_id": 301,
                        "terminal": False,
                        "latest_step": 3000,
                        "expected_step": 4608,
                        "training_passes": 7.8125,
                    }
                ]
            }
        }
    }
    stage_b = {
        "pantry_plan": (
            {"jobs": {"verified_first_global_replay_canonical/s43": 302}},
            {"status": "pass"},
        )
    }
    cells = audit.campaign_cells(stage_a, stage_b)
    assert cells == [
        {
            "domain": "mathir", "arm": "grpo", "seed": 47,
            "job_id": 301, "state": "in_progress", "latest_step": 3000,
            "expected_step": 4608, "training_passes": 7.8125,
            "source": "stage_a_audit",
        },
        {
            "domain": "pantry_plan",
            "arm": "verified_first_global_replay_canonical",
            "seed": 43,
            "job_id": 302,
            "state": "audited_terminal",
            "source": "stage_b_identity_and_audit",
        },
    ]


