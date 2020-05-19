from src.similarity.datedistance import DateDistance
from datetime import timedelta
from assertpy import assert_that


def test_datedistance():
    dd = DateDistance()
    assert_that(1).is_equal_to(dd.get_distance("2000-10-09", "2000-10-08"))
    assert_that(1).is_equal_to(dd.get_distance("2000-10-08", "2000-10-09"))

    assert_that(0).is_equal_to(dd.get_distance("2000-10-09", "2000-10-09"))
    assert_that(0).is_equal_to(dd.get_distance("2000-10", "2000-10"))
    assert_that(0).is_equal_to(dd.get_distance("2000", "2000"))
    assert_that(0).is_equal_to(dd.get_distance("2000", "2000-01-01"))
    assert_that(0).is_equal_to(dd.get_distance("2000-01", "2000-01-01"))

    assert_that(31).is_equal_to(dd.get_distance("2000-01", "2000-02"))

    assert_that(366).is_equal_to(dd.get_distance("2000", "2001"))

    assert_that(timedelta.max).is_equal_to(dd.get_distance("-0006", "2001"))
