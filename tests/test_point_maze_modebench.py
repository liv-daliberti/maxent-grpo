from oat_drgrpo.maze_modebench import POINT_ACTIONS, POINT_MAZE_VERIFIER


def test_point_maze_public_contract_is_available() -> None:
    assert POINT_MAZE_VERIFIER == "point_maze_action_program"
    assert tuple(POINT_ACTIONS) == (
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SW",
        "W",
        "NW",
        "COAST",
    )

