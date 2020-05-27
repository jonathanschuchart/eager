from typing import List, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsTransformer, DistanceMetric
from openea.modules.load.kgs import KGs
from similarity.measure_finding import get_measures
from multiprocessing import Pool

# inherited objects for read-access
embedding_calc_init = {}


def init_calc_from_embedding(
    embedding: np.ndarray,
    kgs: KGs,
    metric: str,
    neigh_dist: np.array,
    neigh_ind: np.array,
):
    embedding_calc_init["embedding"] = embedding
    embedding_calc_init["kgs"] = kgs
    embedding_calc_init["metric"] = metric
    embedding_calc_init["neigh_dist"] = neigh_dist
    embedding_calc_init["neigh_ind"] = neigh_ind


def _parallel_calc_from_embedding_function(it_tup):
    i, n = it_tup
    embedding = embedding_calc_init["embedding"]
    kgs = embedding_calc_init["kgs"]
    metric = embedding_calc_init["metric"]
    neigh_dist = embedding_calc_init["neigh_dist"]
    neigh_ind = embedding_calc_init["neigh_ind"]
    similarities = dict()
    for n, distance in zip(neigh_ind[i], neigh_dist[i]):
        if not i == n and not ((i, n) in similarities or (n, i) in similarities):
            similarities[(i, n)] = _calculate_attribute_sims(kgs, i, n)
            similarities[(i, n)][metric] = distance
    return similarities


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
    with Pool(
        initializer=init_calc_from_embedding,
        initargs=(embedding, kgs, metric, neigh_dist, neigh_ind),
    ) as pool:
        sim_dicts = pool.map(
            _parallel_calc_from_embedding_function, enumerate(neigh_ind)
        )
    for s in sim_dicts:
        similarities.update(s)
    return similarities


# inherited objects for read-access
training_calc_init = {}


def init_calc_from_training(
    embedding: np.ndarray, kgs: KGs, dist_metric: DistanceMetric, metric: str
):
    training_calc_init["embedding"] = embedding
    training_calc_init["kgs"] = kgs
    training_calc_init["dist_metric"] = dist_metric
    training_calc_init["metric"] = metric


def _parallel_calc_with_training_function(links):
    embedding = training_calc_init["embedding"]
    kgs = training_calc_init["kgs"]
    dist_metric = training_calc_init["dist_metric"]
    metric = training_calc_init["metric"]
    similarities = dict()
    # TODO one unnecessary comparison? But probably this is not even computed
    emb_slice = [embedding[int(links[0])], embedding[int(links[1])]]
    # pairwise returns 2d array, but we just want the distance
    distance = dist_metric.pairwise(emb_slice)[0][1]
    similarities[(links[0], links[1])] = _calculate_attribute_sims(
        kgs, links[0], links[1]
    )
    similarities[(links[0], links[1])][metric] = distance
    similarities[(links[0], links[1])]["label"] = links[2]
    return similarities


def calculate_from_embeddings_with_training(
    embedding: np.ndarray, links: List[Tuple[int, int, int]], kgs: KGs, metric: str
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
    with Pool(
        initializer=init_calc_from_training,
        initargs=(embedding, kgs, dist_metric, metric),
    ) as pool:
        sim_dicts = pool.map(_parallel_calc_with_training_function, links)
    for s in sim_dicts:
        similarities.update(s)
    return similarities


def calculate_on_demand(embeddings, pair, kgs, metric):
    dist_metric = DistanceMetric.get_metric(metric)
    emb_slice = [embeddings[int(pair[0])], embeddings[int(pair[1])]]
    distance = dist_metric.pairwise(emb_slice)[0][1]
    result = _calculate_attribute_sims(kgs, pair[0], pair[1])
    result[metric] = distance
    return result


def _calculate_attribute_sims(kgs: KGs, e1_index: np.int64, e2_index: np.int64):
    values = dict()
    e1_attrs = _get_attrs(kgs, e1_index)
    e2_attrs = _get_attrs(kgs, e2_index)
    if e1_attrs is not None and e2_attrs is not None:
        for k1, k2 in align_attributes(e1_attrs, e2_attrs, False):
            key = str(k1) + ":" + str(k2) if k1 < k2 else str(k2) + ":" + str(k1)
            # TODO for now k1 and k2 will have the same type, but this might change
            for name, measure in get_measures(e1_attrs[k1]).items():
                values[name + "." + key] = _get_comparison_value(
                    e1_attrs[k1], e2_attrs[k2], measure
                )
    return values


def align_attributes(e1_attrs: dict, e2_attrs: dict, only_trivial=True):
    """
    Aligns the given attributes.
    :param e1_attrs: attributes of entity 1
    :param e2_attrs: attributes of entity 2
    :param only_trivial: if true only return alignment for attrinutes with same id
    :return: tuples of attribute indices
    """
    # add common keys
    trivial = [(k, k) for k in set.intersection(set(e1_attrs), set(e2_attrs))]
    if only_trivial:
        return trivial

    # TODO enhance for more alignments e.g. by type
    aligned = []
    for k1, v1 in e1_attrs.items():
        # already found best alignment
        if k1 in trivial:
            continue
        for k2, v2 in e2_attrs.items():
            # already found best alignment
            if k2 in trivial:
                continue
            if k1 == k2:
                aligned.append((k1, k2))
            elif v1 == v2:
                aligned.append((k1, k2))
            elif "^^" in v1 and "^^" in v2:
                if v1.split("^^")[1] == v2.split("^^")[1]:
                    aligned.append((k1, k2))
            elif "^^" not in v1 and "^^" not in v2:
                aligned.append((k1, k2))
    aligned.extend(trivial)
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


def _get_attrs(kgs: KGs, index) -> dict:
    if index not in kgs.kg1.av_dict:
        if index not in kgs.kg2.av_dict:
            return None
        else:
            attributes = kgs.kg2.av_dict[index]
    else:
        attributes = kgs.kg1.av_dict[index]
    return dict((k, v) for k, v in attributes)
