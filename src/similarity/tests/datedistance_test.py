from similarity.datedistance import DateDistance


def test_datedistance():
    dd = DateDistance()
    assert 1 == dd.get_distance("2000-10-09", "2000-10-08")
    assert 1 == dd.get_distance("2000-10-08", "2000-10-09")

    assert 0 == dd.get_distance("2000-10-09", "2000-10-09")
    assert 0 == dd.get_distance("2000-10", "2000-10")
    assert 0 == dd.get_distance("2000", "2000")
    assert 0 == dd.get_distance("2000", "2000-01-01")
    assert 0 == dd.get_distance("2000-01", "2000-01-01")

    assert 31 == dd.get_distance("2000-01", "2000-02")

    assert 366 == dd.get_distance("2000", "2001")
