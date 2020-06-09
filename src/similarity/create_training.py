import numpy as np
import pandas as pd
import yaml
import sys
import os
from datetime import datetime
from typing import List, Tuple
from openea.modules.load.kgs import KGs, read_kgs_from_folder
from similarity.similarities import (
    calculate_from_embeddings_with_training,
    calculate_from_embeddings,
)
from dataset.dataset import sample_negative
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
    df: pd.DataFrame, cols: List[str], min_max: MinMaxScaler = None
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    min_max = min_max or MinMaxScaler().fit(df[cols])
    df[cols] = min_max.transform(df[cols])
    df[cols] = 1 - df[cols]
    return df, min_max


def _get_columns_to_normalize(df, measurenames):
    wanted_cols = []
    for col in df.columns:
        for m in measurenames:
            if m in col:
                wanted_cols.append(col)
    return wanted_cols


def create_labeled_similarity_frame(
    similarities: dict, min_max: MinMaxScaler = None
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Creates pandas DataFrame with the similarities and labels (if labels for tuple are present)
    Distances will be normalized to similarities
    :param similarities: dictionary of dictionaries of similarities per entity tuple
    :return: SparseDataFrame with labels if available
    """
    # create similarity frame
    sim_frame = pd.DataFrame.from_dict(similarities, orient="index", dtype="float32")
    print("Normalizing dataframe...")
    cols = _get_columns_to_normalize(sim_frame, cols_to_normalize)
    df, min_max = create_normalized_sim_from_dist_cols(sim_frame, cols, min_max)
    return df, min_max


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


def create_similarity_frame_on_demand(
    embeddings: np.ndarray,
    tup: Tuple,
    kgs: KGs,
    min_max: MinMaxScaler,
    metric="euclidean",
):
    similarities = calculate_from_embeddings_with_training(
        embeddings, [tup], kgs, metric
    )
    return create_labeled_similarity_frame(similarities, min_max)


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
        similarities.update({k: dict(similarities.get(k, {}), **v)
                             for k, v in similarities_embedding.items()})
    print("Creating DataFrame")
    return create_labeled_similarity_frame(similarities)


if __name__ == "__main__":
    startTime = datetime.now()
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
    nearest_neighbors = (
        arguments["nearest_neighbors"] if "nearest_neighbors" in arguments else -1
    )
    metric = arguments["metric"] if "metric" in arguments else "euclidean"
    link_names = ["test_links", "train_links", "valid_links"]
    result_frames = []

    for wanted_links in link_names:
        print(f"Prepare creation of similarities for {wanted_links}")
        labeled_tuples = read_examples(
            training_data_folder + division + wanted_links, kg1_entities, kg2_entities,
        )

        if "sample_negative" in arguments and arguments["sample_negative"]:
            labeled_tuples.extend(sample_negative(labeled_tuples))

        # feature vector creation
        df, _ = create_feature_similarity_frame(
            embedding, labeled_tuples, kgs, nearest_neighbors, metric, only_training,
        )
        if "drop_na_threshold" in arguments:
            thresh = arguments["drop_na_threshold"]
            print(f"dropping columns with less than {thresh} non-na values")
            df = df.dropna(axis=1, how="all", thresh=thresh)
        result_frames.append(df)
        print(f"Created similarity frame for {wanted_links}")
    # adjust headers
    common_header = (
        set(result_frames[0].columns)
        .union(set(result_frames[1].columns))
        .union(set(result_frames[2].columns))
    )
    for name, df in zip(link_names, result_frames):
        missing_cols = common_header - set(df.columns)
        # create empty matrix
        empties = (np.full(len(missing_cols), np.nan),) * len(df)
        # create empty frame with column names
        missing_frame = pd.DataFrame(empties, columns=missing_cols)
        missing_frame.index = df.index
        df = pd.concat([df, missing_frame], axis=1).sort_index(axis=1)
        out_file_path = output + "/" + name + ".pkl"
        df.to_pickle(out_file_path, protocol=2)
        print(f"Wrote similarity frame for {name} to {out_file_path}")
    duration = datetime.now() - startTime
    print(
        f"Creation of dataframes took {duration.seconds//60} minutes and {duration.seconds%60} seconds"
    )
