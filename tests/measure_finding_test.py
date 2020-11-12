import src.similarity.measure_finding as mf
from attribute_features import CartesianCombination
from distance_measures import DateDistance
from similarity_measures import (
    NumberSimilarity,
    Levenshtein,
    TriGram,
    GeneralizedJaccard,
)


def test_measure_finding():
    attr_comb = CartesianCombination(
        None,
        [NumberSimilarity()],
        [DateDistance()],
        [Levenshtein(), TriGram(), GeneralizedJaccard()],
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"2.88E9"^^<http://dbpedia.org/datatype/australianDollar>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"2"^^<http://dbpedia.org/datatype/bangladeshiTaka>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"1243.23"^^<http://dbpedia.org/datatype/canadianDollar>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"12"^^<http://dbpedia.org/datatype/colombianPeso>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"123"^^<http://dbpedia.org/datatype/croatianKuna>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"123"^^<http://dbpedia.org/datatype/euro>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"123"^^<http://dbpedia.org/datatype/hongKongDollar>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"123"^^<http://dbpedia.org/datatype/indianRupee>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"123"^^<http://dbpedia.org/datatype/japaneseYen>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"123"^^<http://dbpedia.org/datatype/norwegianKrone>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"123"^^<http://dbpedia.org/datatype/poundSterling>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"123"^^<http://dbpedia.org/datatype/russianRouble>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"123"^^<http://dbpedia.org/datatype/southKoreanWon>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"123"^^<http://dbpedia.org/datatype/swedishKrona>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"123"^^<http://dbpedia.org/datatype/swissFranc>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"123"^^<http://dbpedia.org/datatype/usDollar>'
    )
    # no explicit type
    assert attr_comb.string_measures == attr_comb._get_type_measures("test")

    # dates
    assert attr_comb.date_measures == attr_comb._get_type_measures(
        '"2000-10-01"^^<http://www.w3.org/2001/XMLSchema#date>'
    )
    assert attr_comb.date_measures == attr_comb._get_type_measures(
        '"1999"^^<http://www.w3.org/2001/XMLSchema#gYear>'
    )
    assert attr_comb.date_measures == attr_comb._get_type_measures(
        '"2001-02"^^<http://www.w3.org/2001/XMLSchema#gYearMonth>'
    )

    # numbers
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"1.0"^^<http://www.w3.org/2001/XMLSchema#double>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"1.0"^^<http://www.w3.org/2001/XMLSchema#decimal>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"1.0"^^<http://www.w3.org/2001/XMLSchema#float>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"1"^^<http://www.w3.org/2001/XMLSchema#integer>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"1"^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>'
    )
    assert attr_comb.number_measures == attr_comb._get_type_measures(
        '"1"^^<http://www.w3.org/2001/XMLSchema#positiveInteger>'
    )
