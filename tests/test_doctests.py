import pytest  # noqa
import doctest
import nlpia2
import nlpia2.ch06

# import nlpia2.ch06.spell


# def test_ch06_spell_doctests():
#     results = doctest.testmod(nlpia2.ch06.spell, optionflags=(doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE))
#     assert results.failed == 0


def test_nlpia2_doctests():
    results = doctest.testmod(nlpia2, optionflags=(doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE))
    assert results.failed == 0

