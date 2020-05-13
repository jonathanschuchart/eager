import numpy as np
from sklearn.neighbors import KNeighborsTransformer
from openea.modules.load.kgs import KGs, read_kgs_from_folder
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from py_stringmatching.similarity_measure.generalized_jaccard import GeneralizedJaccard
from py_stringmatching.similarity_measure.dice import Dice
from py_stringmatching.tokenizer.alphanumeric_tokenizer import AlphanumericTokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from measure_finding import get_measures


def calculate_from_embeddings(
    embedding: np.ndarray, kgs: KGs, n_neighbors: int, metric: str
):
    neigh = KNeighborsTransformer(
        mode="distance", n_neighbors=n_neighbors, metric=metric, n_jobs=-1
    )
    neigh.fit(embedding)
    neigh_dist, neigh_ind = neigh.kneighbors(embedding, return_distance=True)
    similarities = dict()
    # TODO parallelize
    i = 0
    for n in neigh_ind[i]:
        if not i == n and not ((i, n) in similarities or (n, i) in similarities):
            similarities[(i, n)] = calculate_attribute_sims(kgs, i, n)
    print(similarities)


def calculate_attribute_sims(kgs: KGs, e1_index: np.int64, e2_index: np.int64):
    values = dict()
    e1_attrs = get_attrs(kgs, e1_index)
    e2_attrs = get_attrs(kgs, e2_index)
    for k1, k2 in align_attributes(e1_attrs, e2_attrs):
        key = str(k1) + ":" + str(k2) if k1 < k2 else str(k2) + ":" + str(k1)
        # TODO for now k1 and k2 will have the same type, but this might change
        for name, measure in get_measures(e1_attrs[k1]).items():
            values[name + "." + key] = get_comparison_value(
                e1_attrs[k1], e2_attrs[k2], measure
            )
    return values


def _get_comparison_value(attr1: str, attr2: str, measure):
    if isinstance(measure, tuple):
        measure_func = measure[0]
        tokenizer = measure[1]
        if hasattr(measure_func, "get_sim_score"):
            return measure_func.get_sim_score(
                tokenizer.tokenize(attr1), tokenizer.tokenize(attr2)
            )
        else:
            return measure_func.get_distance(
                tokenizer.tokenize(attr1), tokenizer.tokenize(attr2)
            )
    if hasattr(measure, "get_sim_score"):
        return measure.get_sim_score(attr1, attr2)
    else:
        return measure.get_distance(attr1, attr2)


def align_attributes(e1_attrs: set, e2_attrs: set):
    # add common keys
    aligned = [(k, k) for k in set.intersection(set(e1_attrs), set(e2_attrs))]
    # TODO enhance for more alignments e.g. by type
    return aligned


def _get_attrs(kgs: KGs, index: np.int64) -> dict:
    if index in kgs.kg1.av_dict:
        attributes = kgs.kg1.av_dict[index]
    else:
        attributes = kgs.kg2.av_dict[index]
    return dict((k, v) for k, v in attributes)


# embedding = np.load(
#     "/tmp/output/results/GCN_Align/D_W_15K_V1/721_5fold/1/20200513192316/ent_embeds.npy"
# )
# kgs = read_kgs_from_folder(
#     "/home/dobraczka/Downloads/git/OpenEA/datasets/D_Y_15K_V1/",
#     "721_5fold/1/",
#     "mapping",
#     True,
#     remove_unlinked=False,
# )
# print(calculate_attribute_sims(kgs, 0, 12))
