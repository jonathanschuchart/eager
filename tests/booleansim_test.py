from src.similarity.booleansim import BooleanSimilarity


def test_boolean_sim():
    bs = BooleanSimilarity()
    assert 1.0 == bs.get_sim_score("true", "true")
    assert 0.0 == bs.get_sim_score("true", "false")
    assert 0.0 == bs.get_sim_score("false", "true")
    assert 1.0 == bs.get_sim_score("false", "false")
