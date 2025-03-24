import pytest

from microtorch.tensor.utils import identify_broadcasting_dimensions


def test_identical_shapes():
    s1 = (3, 4, 5)
    s2 = (3, 4, 5)
    result = identify_broadcasting_dimensions(s1, s2)
    assert result == ((), ())


def test_broadcasting_on_left_operand():
    s1 = (4,)
    s2 = (2, 3, 4)
    result = identify_broadcasting_dimensions(s1, s2)
    assert result == ((0, 1), ())


def test_broadcasting_on_right_operand():
    s1 = (2, 3, 4)
    s2 = (4,)
    result = identify_broadcasting_dimensions(s1, s2)
    assert result == ((), (0, 1))


def test_broadcasting_on_both_operands():
    s1 = (2, 1, 4)
    s2 = (2, 3, 1)
    result = identify_broadcasting_dimensions(s1, s2)
    assert result == ((1,), (2,))


def test_broadcasting_on_multiple_dimensions():
    s1 = (2, 1, 3, 4, 5)
    s2 = (2, 3, 1, 1, 5)
    result = identify_broadcasting_dimensions(s1, s2)
    assert result == ((1,), (2, 3))


def test_invalid_broadcasting_1():
    s1 = (2, 3)
    s2 = (2, 4)
    with pytest.raises(ValueError):
        identify_broadcasting_dimensions(s1, s2)


def test_invalid_broadcasting_2():
    s1 = (2, 3)
    s2 = (4, 3)
    with pytest.raises(ValueError):
        identify_broadcasting_dimensions(s1, s2)
