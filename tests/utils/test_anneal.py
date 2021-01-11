from numpy.testing import assert_almost_equal
from dsvae.utils import linear_anneal


def test_linear_anneal():
    test_cases = [(20, 100, 0.8), (0, 50, 1.0), (99, 100, 0.01), (102, 100, 0.0)]

    for case in test_cases:
        got = linear_anneal(case[0], case[1])
        assert_almost_equal(got, case[2])
