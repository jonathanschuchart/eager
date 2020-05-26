import numpy as np
import pandas as pd
import yaml
import sys
import os
from typing import List
from openea.modules.load.kgs import KGs, read_kgs_from_folder
from similarity.similarities import (
    calculate_from_embeddings_with_training,
    calculate_from_embeddings,
)
from main import sample_negative
from sklearn.preprocessing import MinMaxScaler

cols_to_normalize = ["Date", "euclidean"]


def _get_feature_name_list(similarities: dict) -> List:
    feature_set = set()
    for k, v in similarities.items():
        feature_set.update(v.keys())
    feature_names = list(feature_set)
    feature_names.sort()
    return feature_names


def _create_tuple_string(tup) -> str:
    if tup[0] < tup[1]:
        return str(tup[0]) + "," + str(tup[1])
    else:
        return str(tup[1]) + "," + str(tup[0])


def create_normalized_sim_from_dist_cols(
    df: pd.DataFrame, cols: List[str]
) -> pd.DataFrame:
    df[cols] = MinMaxScaler().fit_transform(df[cols])
    df[cols] = 1 - df[cols]
    return df


def _get_columns_to_normalize(df, measurenames):
    wanted_cols = []
    for col in df.columns:
        for m in measurenames:
            if m in col:
                wanted_cols.append(col)
    return wanted_cols


def create_labeled_similarity_frame(
    similarities: dict, labeled_tuples: List
) -> pd.SparseDataFrame:
    """
    Creates pandas DataFrame with the similarities and labels (if labels for tuple are present)
    Distances will be normalized to similarities
    :param similarities: dictionary of dictionaries of similarities per entity tuple
    :param labeled_tuples: list of triples with the first two entries denoting the entity tuples and the last the label
    :return: SparseDataFrame with labels if available
    """
    tmp_lab_dict = dict()
    for t in labeled_tuples:
        tmp_lab_dict[(t[0], t[1])] = t[2]
    sim_frame = pd.SparseDataFrame.from_dict(
        similarities, orient="index", dtype="float32"
    )
    lab_frame = pd.Series(tmp_lab_dict).to_frame("label")
    sim_frame = sim_frame.merge(
        lab_frame, left_index=True, right_index=True, how="outer"
    )
    return create_normalized_sim_from_dist_cols(
        sim_frame, _get_columns_to_normalize(sim_frame, cols_to_normalize)
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


# TODO this could already be loaded in kgs?
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


def create_feature_similarity_frame(
    embedding: np.array,
    labeled_tuples: List,
    kgs: KGs,
    n_neighbors=5,
    metric="euclidean",
    only_training=False,
):
    similarities = calculate_from_embeddings_with_training(
        embedding, labeled_tuples, kgs, metric
    )
    print("Finished calculation from given links")
    if not only_training:
        similarities_embedding = calculate_from_embeddings(
            embedding, kgs, n_neighbors, metric
        )
        print("Finished calculation from nearest neighbors")
        # merge both
        similarities.update(similarities_embedding)
    return create_labeled_similarity_frame(similarities, labeled_tuples)


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
    # dont use nearest neighbors if true
    only_training = (
        arguments["only_training"] if "only_training" in arguments else False
    )
    metric = arguments["metric"] if "metric" in arguments else "euclidean"
    link_names = ["test_links", "train_links", "valid_links"]

    for wanted_links in link_names:
        print(f"Prepare creation of similarities for {wanted_links}")
        labeled_tuples = read_examples(
            training_data_folder + division + wanted_links, kg1_entities, kg2_entities,
        )

        if "sample_negative" in arguments and arguments["sample_negative"]:
            labeled_tuples.extend(sample_negative(labeled_tuples))

        # feature vector creation
        features_frame = create_feature_similarity_frame(
            embedding,
            labeled_tuples,
            kgs,
            arguments["nearest_neighbors"],
            metric,
            only_training,
        )
        out_file_path = output + "/" + wanted_links + ".pkl"
        features_frame.to_pickle(out_file_path, protocol=2)
        print(f"Wrote similarity frame for {wanted_links} to {out_file_path}")
