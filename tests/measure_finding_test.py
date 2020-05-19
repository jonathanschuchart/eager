import src.similarity.measure_finding as mf


def test_measure_finding():
    assert mf.number_measures == mf.get_measures(
        '"2.88E9"^^<http://dbpedia.org/datatype/australianDollar>'
    )
    assert mf.number_measures == mf.get_measures(
        '"2"^^<http://dbpedia.org/datatype/bangladeshiTaka>'
    )
    assert mf.number_measures == mf.get_measures(
        '"1243.23"^^<http://dbpedia.org/datatype/canadianDollar>'
    )
    assert mf.number_measures == mf.get_measures(
        '"12"^^<http://dbpedia.org/datatype/colombianPeso>'
    )
    assert mf.number_measures == mf.get_measures(
        '"123"^^<http://dbpedia.org/datatype/croatianKuna>'
    )
    assert mf.number_measures == mf.get_measures(
        '"123"^^<http://dbpedia.org/datatype/euro>'
    )
    assert mf.number_measures == mf.get_measures(
        '"123"^^<http://dbpedia.org/datatype/hongKongDollar>'
    )
    assert mf.number_measures == mf.get_measures(
        '"123"^^<http://dbpedia.org/datatype/indianRupee>'
    )
    assert mf.number_measures == mf.get_measures(
        '"123"^^<http://dbpedia.org/datatype/japaneseYen>'
    )
    assert mf.number_measures == mf.get_measures(
        '"123"^^<http://dbpedia.org/datatype/norwegianKrone>'
    )
    assert mf.number_measures == mf.get_measures(
        '"123"^^<http://dbpedia.org/datatype/poundSterling>'
    )
    assert mf.number_measures == mf.get_measures(
        '"123"^^<http://dbpedia.org/datatype/russianRouble>'
    )
    assert mf.number_measures == mf.get_measures(
        '"123"^^<http://dbpedia.org/datatype/southKoreanWon>'
    )
    assert mf.number_measures == mf.get_measures(
        '"123"^^<http://dbpedia.org/datatype/swedishKrona>'
    )
    assert mf.number_measures == mf.get_measures(
        '"123"^^<http://dbpedia.org/datatype/swissFranc>'
    )
    assert mf.number_measures == mf.get_measures(
        '"123"^^<http://dbpedia.org/datatype/usDollar>'
    )
    # no explicit type
    assert mf.string_similarity_measures == mf.get_measures("test")

    # dates
    assert mf.date_measures == mf.get_measures(
        '"2000-10-01"^^<http://www.w3.org/2001/XMLSchema#date>'
    )
    assert mf.date_measures == mf.get_measures(
        '"1999"^^<http://www.w3.org/2001/XMLSchema#gYear>'
    )
    assert mf.date_measures == mf.get_measures(
        '"2001-02"^^<http://www.w3.org/2001/XMLSchema#gYearMonth>'
    )

    # numbers
    assert mf.number_measures == mf.get_measures(
        '"1.0"^^<http://www.w3.org/2001/XMLSchema#double>'
    )
    assert mf.number_measures == mf.get_measures(
        '"1.0"^^<http://www.w3.org/2001/XMLSchema#decimal>'
    )
    assert mf.number_measures == mf.get_measures(
        '"1.0"^^<http://www.w3.org/2001/XMLSchema#float>'
    )
    assert mf.number_measures == mf.get_measures(
        '"1"^^<http://www.w3.org/2001/XMLSchema#integer>'
    )
    assert mf.number_measures == mf.get_measures(
        '"1"^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>'
    )
    assert mf.number_measures == mf.get_measures(
        '"1"^^<http://www.w3.org/2001/XMLSchema#positiveInteger>'
    )
