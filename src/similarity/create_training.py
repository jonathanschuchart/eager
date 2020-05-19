import numpy as np
import yaml
import os
import sys
from typing import List
from openea.modules.load.kgs import KGs, read_kgs_from_folder
from similarity.similarities import (
    calculate_from_embeddings_with_training,
    calculate_from_embeddings,
)


def _get_feature_name_list(similarities: dict) -> List:
    feature_set = set()
    for k, v in similarities.items():
        feature_set.update(v.keys())
    feature_names = list(feature_set)
    feature_names.sort()
    return feature_names


def create_from_similarities(similarities: dict, labeled_tuples: List):
    feature_names = _get_feature_name_list(similarities)
    # calculate features for training
    features = []
    labels = []
    used = set()
    for l in labeled_tuples:
        tuple_sims = similarities[(l[0], l[1])]
        used.add((l[0], l[1]))
        tuple_features = []
        for name in feature_names:
            if name in tuple_sims:
                tuple_features.append(tuple_sims[name])
            else:
                tuple_features.append(np.nan)
        features.append(tuple_features)
        labels.append(l[2])

    # calculate rest
    features_unlabeled = []
    for k, tuple_sims in similarities.items():
        if not k in used:
            tuple_features = []
            for name in feature_names:
                if name in tuple_sims:
                    tuple_features.append(tuple_sims[name])
                else:
                    tuple_features.append(np.nan)
            features_unlabeled.append(tuple_features)

    return (
        np.array(features),
        np.array(labels),
        feature_names,
        np.array(features_unlabeled),
    )


def read_entity_ids(path: str) -> dict:
    entity_dict = dict()
    with open(path, "r") as f:
        for line in f:
            e_id = line.strip().split("\t")
            entity_dict[e_id[0]] = e_id[1]
    return entity_dict


def _get_entity_id_by_url(url: str, kg1_entities: dict, kg2_entities: dict):
    if url in kg1_entities:
        return int(kg1_entities[url])
    return int(kg2_entities[url])


def read_examples(path: str, kg1_entities: dict, kg2_entities: dict):
    # TODO negative examples
    labeled_tuples = []
    with open(path, "r") as f:
        for line in f:
            tuples = line.strip().split("\t")
            left = _get_entity_id_by_url(tuples[0], kg1_entities, kg2_entities)
            right = _get_entity_id_by_url(tuples[1], kg1_entities, kg2_entities)
            labeled_tuples.append((left, right, 1))
    return labeled_tuples


def create_feature_vectors(
    embedding: np.array,
    labeled_tuples: List,
    kgs: KGs,
    n_neighbors: int,
    metric="euclidean",
):
    similarities = calculate_from_embeddings(embedding, kgs, n_neighbors, metric)
    print("Finished calculation from nearest neighbors")
    similarities_training = calculate_from_embeddings_with_training(
        embedding, labeled_tuples, kgs, metric
    )
    print("Finished calculation from training")
    # merge both
    similarities.update(similarities_training)
    return create_from_similarities(similarities, labeled_tuples)


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        arguments = yaml.load(f, yaml.FullLoader)

    output = arguments["output"]
    if not os.path.exists(output):
        os.makedirs(output)

    # Load all necessary files
    embedding = np.load(arguments["embedding"])
    kg1_entities = read_entity_ids(arguments["kg1_entities"])
    kg2_entities = read_entity_ids(arguments["kg2_entities"])
    training_data_folder = arguments["training_data"]
    division = arguments["dataset_division"]
    remove_unlinked = False
    kgs = read_kgs_from_folder(
        training_data_folder,
        division,
        arguments["alignment_module"],
        arguments["ordered"],
        remove_unlinked=remove_unlinked,
    )
    labeled_tuples = read_examples(
        training_data_folder + division + "train_links", kg1_entities, kg2_entities,
    )

    # feature vector creation
    features, labels, feature_names, features_unlabeled = create_feature_vectors(
        embedding, labeled_tuples, kgs, arguments["nearest_neighbors"]
    )

    # write
    feature_out = output + "/features.npy"
    label_out = output + "/labels.npy"
    feat_name_out = output + "/feature_names.txt"
    feature_unlabeled_out = output + "/features_unlabeled.npy"

    np.save(feature_out, features)
    np.save(feature_unlabeled_out, features_unlabeled)
    print(f"Wrote {len(features)} features to {feature_out}")
    print(f"Wrote {len(features_unlabeled)} features to {feature_unlabeled_out}")
    print(f"Wrote {len(labels)} labels to {label_out}")
    np.save(label_out, labels)
    with open(feat_name_out, "w") as f:
        for n in feature_names:
            f.write("%s\n" % n)
    print(f"Wrote {len(feature_names)} features to {feat_name_out}")
