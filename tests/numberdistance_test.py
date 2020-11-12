from src.similarity_measures import NumberSimilarity


def test_numberdistance():
    nd = NumberSimilarity()
    assert 0 == nd("0", "0")
    assert 0 == nd("0.0", "0.0")
    assert 1 == nd("1.0", "0.0")
    assert 1 == nd("1", "0")
    assert 1 == nd("0", "1")
    assert 1 == nd("0", "-1")
