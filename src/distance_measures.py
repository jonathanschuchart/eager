from abc import ABC, abstractmethod
from datetime import timedelta, date
from typing import Union

from py_stringmatching import utils
from scipy.spatial.distance import euclidean
from sentence_transformers import SentenceTransformer
import numpy as np


class DistanceMeasure(ABC):
    """
    A measure representing the distance between two values.
    These can be 2 single attribute values (interpreted as number/date/string accordingly)
    or numpy arrays if used as embedding distance.

    Beware that any results computed from a subclass of this will be normalized to a
    [0, 1] similarity value after the fact for classification.
    To avoid this normalization, see SimilarityMeasure.
    """

    @abstractmethod
    def __call__(self, v1, v2) -> Union[float, np.ndarray]:
        pass


class NumberDistance(DistanceMeasure):
    def __call__(self, v1, v2) -> float:
        return abs(float(v1) - float(v2))


class DateDistance(DistanceMeasure):
    """
    Used to calculate the distance between datestrings
    """

    @staticmethod
    def _create_date(date_string: str):
        try:
            if not "-" in date_string:
                # only year
                return date(int(date_string), 1, 1)
            if date_string.count("-") == 2:
                year, month, day = date_string.split("-")
                return date(int(year), int(month), int(day))
            else:
                year, month = date_string.split("-")
                return date(int(year), int(month), 1)
        except ValueError as e:
            return None

    def __call__(self, date_string1: str, date_string2: str) -> int:
        """
        Dates will be parsed to datetime.date and the absolute difference in dates is returned
        :param date_string1: one date as string
        :param date_string2: other date string
        :return: difference in days
        """
        date1 = DateDistance._create_date(date_string1)
        date2 = DateDistance._create_date(date_string2)
        if date1 is None or date2 is None:
            return timedelta.max.days
        return abs(date1 - date2).days


class BertFeatureDistance(DistanceMeasure):
    def __init__(self, bert_key="distilbert-multilingual-nli-stsb-quora-ranking"):
        super(BertFeatureDistance, self).__init__()
        self.model = SentenceTransformer(bert_key)

    def __call__(self, string1, string2) -> np.ndarray:
        # convert input strings to unicode.
        string1 = utils.convert_to_unicode(string1)
        string2 = utils.convert_to_unicode(string2)
        max_len = max(len(string1), len(string2))
        if max_len == 0:
            return [1.0] * self.model.get_sentence_embedding_dimension()
        e1 = self.model.encode(string1)
        e2 = self.model.encode(string2)
        return np.abs(e1 - e2)


class EmbeddingEuclideanDistance(DistanceMeasure):
    def __call__(self, v1, v2) -> float:
        return euclidean(v1, v2)
