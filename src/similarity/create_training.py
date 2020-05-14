import numpy as np
from typing import List


import pickle


def _get_feature_name_list(similarities: dict) -> List:
    feature_set = set()
    for k, v in similarities.items():
        feature_set.update(v.keys())
    feature_names = list(feature_set)
    feature_names.sort()
    return feature_names


def create_from_similarities(similarities: dict, labeled_tuples: List):
    feature_names = _get_feature_name_list(similarities)
    features = []
    labels = []
    for l in labeled_tuples:
        tuple_sims = similarities[(l[0], l[1])]
        tuple_features = []
        for name in feature_names:
            if name in tuple_sims:
                tuple_features.append(tuple_sims[name])
            else:
                tuple_features.append(np.nan)
        features.append(tuple_features)
        labels.append(l[2])
    return np.array(features), np.array(labels), feature_names


def read_entity_ids(path: str) -> dict:
    entity_dict = dict()
    with open(path, "r") as f:
        for line in f:
            e_id = line.strip().split("\t")
            entity_dict[e_id[1]] = e_id[0]
    return entity_dict


def _get_entity_id_by_url(url: str, kg1_entities: dict, kg2_entities: dict):
    if tuples[0] in kg1_entities:
        return kg1_entities[url]
    return kg2_entities[url]


def read_examples(path: str, kg1_entities: dict, kg2_entities: dict):
    # TODO negative examples
    labeled_tuples = []
    with open(path, "r") as f:
        for line in f:
            tuples = line.strip().split("\t")
            left = _get_entity_id_by_url[tuples[0]]
            right = _get_entity_id_by_url[tuples[1]]
            labeled_tuples.append(left, right, 1)
    return labeled_tuples


embedding = np.load("src/similarity/tests/test_kgs/slice_ent_emb.npy")
kgs = pickle.load(open("src/similarity/tests/test_kgs/kgs.pkl", "rb"))
kg1_entities = read_entity_ids("src/similarity/tests/test_kgs/kg1_ent_ids")
kg2_entities = read_entity_ids("src/similarity/tests/test_kgs/kg2_ent_ids")
