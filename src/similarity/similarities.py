import numpy as np
from sklearn.neighbors import KNeighborsTransformer, DistanceMetric
from openea.modules.load.kgs import KGs, read_kgs_from_folder
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from py_stringmatching.similarity_measure.generalized_jaccard import GeneralizedJaccard
from py_stringmatching.similarity_measure.dice import Dice
from py_stringmatching.tokenizer.alphanumeric_tokenizer import AlphanumericTokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from measure_finding import get_measures


def calculate_from_embeddings(
    embedding: np.ndarray, kgs: KGs, n_neighbors: int, metric: str
) -> dict:
    """
    Uses the given embeddings to find the n-NearestNeighbors of each entity and calculate the similarities.
    The similarities are calculated on the attributes of the corresponding entities
    :param embedding: numpy array with embeddings
    :param kgs: knowledge graphs
    :param n_neighbors: number of nearest neighbors that will be compared
    :param metric: distance metric that will be used to find nearest neighbors
    :return: dictionary with tuples of entity indices that were compared as keys and a dictionary of comparisons as value where the respective keys represent the measure and attribute combination
    """
    neigh = KNeighborsTransformer(
        mode="distance", n_neighbors=n_neighbors, metric=metric, n_jobs=-1
    )
    neigh.fit(embedding)
    neigh_dist, neigh_ind = neigh.kneighbors(embedding, return_distance=True)
    similarities = dict()
    # TODO parallelize
    for i, n in enumerate(neigh_ind):
        for n, distance in zip(neigh_ind[i], neigh_dist[i]):
            if not i == n and not ((i, n) in similarities or (n, i) in similarities):
                similarities[(i, n)] = _calculate_attribute_sims(kgs, i, n)
                similarities[(i, n)][metric] = distance
    return similarities


def calculate_from_embeddings_with_training(
    embedding: np.ndarray, links: tuple, kgs: KGs, metric: str
) -> dict:
    """
    Uses the given embeddings and links to calculate the similarities/distances in
    metric space and on attributes.
    :param embedding: numpy array with embeddings
    :param links: triple of entity ids and 0/1 label
    :param kgs: knowledge graphs
    :param metric: distance metric that will be used to find nearest neighbors
    :return: dictionary with tuples of entity indices that were compared as keys and a dictionary of comparisons as value where the respective keys represent the measure and attribute combination
    """
    dist_metric = DistanceMetric.get_metric(metric)
    similarities = dict()
    # TODO parallelize
    for l in links:
        # TODO one unnecessary comparison? But probably this is not even computed
        emb_slice = [embedding[l[0]], embedding[l[1]]]
        # pairwise returns 2d array, but we just want the distance
        distance = dist_metric.pairwise(emb_slice)[0][1]
        similarities[(l[0], l[1])] = _calculate_attribute_sims(kgs, l[0], l[1])
        similarities[(l[0], l[1])][metric] = distance
    return similarities


def _calculate_attribute_sims(kgs: KGs, e1_index: np.int64, e2_index: np.int64):
    values = dict()
    e1_attrs = _get_attrs(kgs, e1_index)
    e2_attrs = _get_attrs(kgs, e2_index)
    for k1, k2 in align_attributes(e1_attrs, e2_attrs):
        key = str(k1) + ":" + str(k2) if k1 < k2 else str(k2) + ":" + str(k1)
        # TODO for now k1 and k2 will have the same type, but this might change
        for name, measure in get_measures(e1_attrs[k1]).items():
            values[name + "." + key] = _get_comparison_value(
                e1_attrs[k1], e2_attrs[k2], measure
            )
    return values


def align_attributes(e1_attrs: set, e2_attrs: set):
    """
    Aligns the given attributes.
    For now only using common indices of attributes for alignment
    :param e1_attrs: attributes of entity 1
    :param e2_attrs: attributes of entity 2
    :return: tuples of attribute indices
    """
    # add common keys
    aligned = [(k, k) for k in set.intersection(set(e1_attrs), set(e2_attrs))]
    # TODO enhance for more alignments e.g. by type
    return aligned


def _remove_type(attr: str) -> str:
    if "^^" in attr:
        attr = attr.split("^^")[0]
    if attr.startswith('"') and attr.endswith('"'):
        return attr[1:-1]
    return attr


def _get_comparison_value(attr1: str, attr2: str, measure) -> float:
    attr1 = _remove_type(attr1)
    attr2 = _remove_type(attr2)
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


def _get_attrs(kgs: KGs, index: np.int64) -> dict:
    if index in kgs.kg1.av_dict:
        attributes = kgs.kg1.av_dict[index]
    else:
        attributes = kgs.kg2.av_dict[index]
    return dict((k, v) for k, v in attributes)
