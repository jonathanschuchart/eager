from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from py_stringmatching.similarity_measure.generalized_jaccard import GeneralizedJaccard
from py_stringmatching.similarity_measure.dice import Dice
from py_stringmatching.tokenizer.alphanumeric_tokenizer import AlphanumericTokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from numberdistance import NumberDistance
from datedistance import DateDistance
from typing import List

string_similarity_measures = {
    "Lev": Levenshtein(),
    "GenJac": (GeneralizedJaccard(), AlphanumericTokenizer()),
    "Trigram": (Dice(), QgramTokenizer(qval=3)),
}

number_measures = {"NumberDist": NumberDistance()}

date_measures = {"DateDist": DateDistance()}

money_types = {
    "<http://dbpedia.org/datatype/australianDollar>",
    "<http://dbpedia.org/datatype/bangladeshiTaka>",
    "<http://dbpedia.org/datatype/canadianDollar>",
    "<http://dbpedia.org/datatype/colombianPeso>",
    "<http://dbpedia.org/datatype/croatianKuna>",
    "<http://dbpedia.org/datatype/euro>",
    "<http://dbpedia.org/datatype/hongKongDollar>",
    "<http://dbpedia.org/datatype/indianRupee>",
    "<http://dbpedia.org/datatype/japaneseYen>",
    "<http://dbpedia.org/datatype/norwegianKrone>",
    "<http://dbpedia.org/datatype/poundSterling>",
    "<http://dbpedia.org/datatype/russianRouble>",
    "<http://dbpedia.org/datatype/southKoreanWon>",
    "<http://dbpedia.org/datatype/swedishKrona>",
    "<http://dbpedia.org/datatype/swissFranc>",
    "<http://dbpedia.org/datatype/usDollar>",
}

date_types = {
    "<http://www.w3.org/2001/XMLSchema#date>",
    "<http://www.w3.org/2001/XMLSchema#gYear>",
    "<http://www.w3.org/2001/XMLSchema#gYearMonth>",
}

number_types = {
    "<http://www.w3.org/2001/XMLSchema#double>",
    "<http://www.w3.org/2001/XMLSchema#decimal>",
    "<http://www.w3.org/2001/XMLSchema#float>",
    "<http://www.w3.org/2001/XMLSchema#integer>",
    "<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>",
    "<http://www.w3.org/2001/XMLSchema#positiveInteger>",
}


def get_measures(attr: str) -> dict:
    """
    :param attr: attribute string with xml schema datatype
    :return: dict with names of appropriate measures as keys and the corresponding measure as value.
             For string similarity measures the value is a tuple of measure and tokenizer/None depending if it is needed
    """
    # attributes without type are assumed to be str
    if not "^^" in attr:
        return string_similarity_measures
    value, datatype = attr.split("^^")
    if datatype in money_types or datatype in number_types:
        # TODO money conversion probably would be overkill?
        return number_measures
    elif datatype in date_types:
        return date_measures
    else:
        return string_similarity_measures
