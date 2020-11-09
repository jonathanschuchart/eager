from py_stringmatching.similarity_measure.sequence_similarity_measure import (
    SequenceSimilarityMeasure,
)
from py_stringmatching import utils
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity


class BertSimilarity(SequenceSimilarityMeasure):
    def __init__(self):
        super(BertSimilarity, self).__init__()
        self.model = SentenceTransformer(
            "distilbert-multilingual-nli-stsb-quora-ranking"
        )

    def get_raw_score(self, string1, string2):
        utils.sim_check_for_none(string1, string2)

        # convert input to unicode.
        string1 = utils.convert_to_unicode(string1)
        string2 = utils.convert_to_unicode(string2)

        utils.tok_check_for_string_input(string1, string2)

        if utils.sim_check_for_exact_match(string1, string2):
            return 0.0

        e1 = self.model.encode(string1)
        e2 = self.model.encode(string2)

        return cdist(e1, e2, metric="cosine")

    def get_sim_score(self, string1, string2):
        # convert input strings to unicode.
        string1 = utils.convert_to_unicode(string1)
        string2 = utils.convert_to_unicode(string2)

        max_len = max(len(string1), len(string2))
        if max_len == 0:
            return 1.0

        e1 = self.model.encode(string1)
        e2 = self.model.encode(string2)
        return cosine_similarity(e1, e2)
