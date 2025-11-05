import src.utils.import_utils as IU


def test_is_availability_flags_return_bools():
    assert isinstance(IU.is_e2b_available(), bool)
    assert isinstance(IU.is_morph_available(), bool)

