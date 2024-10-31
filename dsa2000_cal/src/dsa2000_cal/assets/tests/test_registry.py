import pytest

from dsa2000_cal.assets.registries import match_func, sort_key_func


@pytest.mark.parametrize('match_pattern,template,overlap', [
    ('a', 'a', 1),
    ('a', 'b', 0),
    ('a', 'aa', 1),
    ('a', 'ab', 1),
    ('a', 'ba', 1),
    ('a', 'bb', 0),
    ('aa', 'aaba', 2),
    ('.*', 'abc', 3),
    ('dsa2000.*', 'dsa2000W', 8)
])
def test_sort_key_func(match_pattern: str, template: str, overlap: int):
    assert sort_key_func(match_pattern, template) == -overlap


@pytest.mark.parametrize('match_pattern,template,match', [
    ('a', 'a', True),
    ('a', 'b', False),
    ('a', 'aa', True),
    ('a', 'ab', True),
    ('a', 'ba', True),
    ('a', 'bb', False),
    ('aa', 'aaba', True),
    ('.*', 'abc', True),
    ('dsa2000.*', 'dsa2000W', True)
])
def test_match_func(match_pattern: str, template: str, match: bool):
    assert match_func(match_pattern, template) == match
