from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np

import py_stringmatching
from py_stringmatching import utils
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityMeasure(ABC):
    """
    A measure representing the similarity between two values.
    These can be 2 single attribute values (interpreted as number/date/string accordingly)
    or numpy arrays if used as embedding similarity.

    Results computed from a subclass of this will be used as-is for classification.
    """

    @abstractmethod
    def __call__(
        self, v1: Union[str, np.ndarray], v2: Union[str, np.ndarray]
    ) -> Union[float, np.ndarray]:
        pass


class Levenshtein(SimilarityMeasure):
    def __init__(self):
        self.lev = py_stringmatching.Levenshtein()

    def __call__(self, v1, v2) -> float:
        return self.lev.get_sim_score(v1, v2)


class GeneralizedJaccard(SimilarityMeasure):
    def __init__(self):
        self.genJac = py_stringmatching.GeneralizedJaccard()
        self.tokenizer = py_stringmatching.AlphanumericTokenizer()

    def __call__(self, v1, v2) -> float:
        return self.genJac.get_sim_score(
            self.tokenizer.tokenize(v1), self.tokenizer.tokenize(v2)
        )


class TriGram(SimilarityMeasure):
    def __init__(self):
        self.dice = py_stringmatching.Dice()
        self.tokenizer = py_stringmatching.QgramTokenizer(qval=3)

    def __call__(self, v1, v2) -> float:
        return self.dice.get_sim_score(
            self.tokenizer.tokenize(v1), self.tokenizer.tokenize(v2)
        )


class BertCosineSimilarity(SimilarityMeasure):
    def __init__(self, bert_key="distilbert-multilingual-nli-stsb-quora-ranking"):
        self.model = SentenceTransformer(bert_key)

    def __call__(self, string1, string2) -> float:
        max_len = max(len(string1), len(string2))
        if max_len == 0:
            return 1.0
        e1, e2 = _bert_embed_strings(self.model, string1, string2)
        return cosine_similarity(e1, e2)


class BertFeatureSimilarity(SimilarityMeasure):
    def __init__(self, bert_key="distilbert-multilingual-nli-stsb-quora-ranking"):
        self.model = SentenceTransformer(bert_key)

    def __call__(self, string1, string2) -> np.ndarray:
        max_len = max(len(string1), len(string2))
        if max_len == 0:
            return np.array([1.0] * self.model.get_sentence_embedding_dimension())
        e1, e2 = _bert_embed_strings(self.model, string1, string2)
        return 1.0 - np.abs(e1 - e2)


class BertConcatenation(SimilarityMeasure):
    def __init__(self, bert_key="distilbert-multilingual-nli-stsb-quora-ranking"):
        self.model = SentenceTransformer(bert_key)

    def __call__(self, string1, string2) -> np.ndarray:
        max_len = max(len(string1), len(string2))
        if max_len == 0:
            return np.array([1.0] * self.model.get_sentence_embedding_dimension())
        e1, e2 = _bert_embed_strings(self.model, string1, string2)
        return np.concatenate((e1, e2))


class EmbeddingCosineSimilarity(SimilarityMeasure):
    def __call__(self, v1, v2) -> float:
        return cosine_similarity(v1, v2)


class EmbeddingConcatenation(SimilarityMeasure):
    def __call__(self, v1, v2) -> np.ndarray:
        return np.concatenate((v1, v2))


def _bert_embed_strings(bert_model, string1, string2):
    string1 = utils.convert_to_unicode(string1)
    string2 = utils.convert_to_unicode(string2)
    e1 = bert_model.encode(string1)
    e2 = bert_model.encode(string2)

    return e1, e2


class NumberSimilarity(SimilarityMeasure):
    def __call__(self, v1, v2) -> float:
        return abs(float(v1) - float(v2))
