from numberdistance import NumberDistance


def test_numberdistance():
    nd = NumberDistance()
    assert 0 == nd.get_distance("0", "0")
    assert 0 == nd.get_distance("0.0", "0.0")
    assert 1 == nd.get_distance("1.0", "0.0")
    assert 1 == nd.get_distance("1", "0")
    assert 1 == nd.get_distance("0", "1")
    assert 1 == nd.get_distance("0", "-1")
