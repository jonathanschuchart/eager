import glob
import sys
import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple
from tqdm import tqdm

sns.set()

##################
# Typing
#################
OWL_THING = "http://www.w3.org/2002/07/owl#Thing"
too_broad = [
    "http://dbpedia.org/ontology/Work",
    "http://dbpedia.org/ontology/Agent",
    OWL_THING,
]
wanted = [
    "http://dbpedia.org/ontology/Person",
    "http://dbpedia.org/ontology/Location",
    "http://dbpedia.org/ontology/Film",
    "http://dbpedia.org/ontology/Organisation",
    "http://dbpedia.org/ontology/MusicalWork",
]
synonym = {"http://dbpedia.org/ontology/Place": "http://dbpedia.org/ontology/Location"}

#############
# General
############
EA_col = "Embedding approach"

ds_name_long = {
    "D": "DBpedia",
    "W": "Wikidata",
    "Y": "Yago",
    "EN": "DBpedia EN",
    "FR": "DBpedia FR",
    "DE": "DBpedia DE",
    "imdb": "IMDB",
    "tmdb": "TMDB",
    "tvdb": "TVDB",
}


def get_id_url_dict(id_file_path: str) -> dict:
    """
    Creates a dict with id of entity as key and url as value
    @param id_file_path: path of entity ids file
    @return: id->url dict
    """
    id_url = dict()
    with open(id_file_path) as in_file:
        for line in in_file:
            line_tuple = line.strip().split("\t")
            id_url[line_tuple[1]] = line_tuple[0]
    return id_url


def read_pred(file_path: str) -> List[List]:
    """
    Reads prediction file to list of lists
    @param file_path: location of prediction file
    @return: predictions
    """
    pred = []
    with open(file_path) as in_file:
        for line in in_file:
            if "left,right" not in line:
                pred.append(line.strip().split(","))
    return pred


def pred_with_url(pred: List[List], dicts: Tuple[dict, dict]) -> List[List]:
    """
    Exchange ids in predictions with urls
    @param pred: predictions
    @param dicts: tuple of id->url dicts, tuple has to be in same order as entities in pred
    """
    urled_pred = []
    for p in pred:
        enriched = []
        enriched.append(dicts[0][p[0]])
        enriched.append(dicts[1][p[1]])
        enriched.append(int(p[2]))
        enriched.append(int(p[3]))
        urled_pred.append(enriched)
    return urled_pred


def read_json(file_path: str) -> dict:
    with open(file_path) as fp:
        types = json.load(fp)
    return types


# def _find_most_common(types: List[str], type_occurences: dict) -> List[str] or str:
#     filtered_types = []
#     for occ in type_occurences:
#         if occ[0] in types:
#             filtered_types.append(occ[0])
#         if len(filtered_types) == most_common:
#             return occ[0]
#     if len(filtered_types) == 0:
#         if "http://www.w3.org/2002/07/owl#Thing" in types:
#             return "http://www.w3.org/2002/07/owl#Thing"
#         else:
#             return "UNKNOWN"
#     return filtered_types


def _find_best_type_from_multiple(types):
    if any(isinstance(i, list) for i in types):
        types = [item for sublist in types for item in sublist]
    fallback = OWL_THING
    for t in types:
        if t in too_broad:
            fallback = t
        elif "Wikidata" not in t and "http://dbpedia.org/ontology/" in t:
            if t in synonym:
                return synonym[t]
            return t
    return fallback


def _find_best_general(types: List[str], superclasses: dict):
    candidates = []
    for t in types:
        if t in wanted:
            return t
        if t in superclasses:
            class_hierarchy = superclasses[t].copy()
            class_hierarchy.reverse()
            if class_hierarchy is None:
                continue
            if len(class_hierarchy) == 0 or (
                len(class_hierarchy) == 1 and class_hierarchy[0][0] in too_broad
            ):
                candidates.insert(0, t)
            else:
                if t not in class_hierarchy:
                    class_hierarchy.append([t])
                candidates.append(_find_best_type_from_multiple(class_hierarchy))
    if len(candidates) == 1:
        return candidates[0]
    return _find_best_type_from_multiple(candidates)


def find_fitting_types(pred: List[List], type_dict: dict, superclasses: dict) -> dict:
    typed_preds = []
    for p in pred:
        enriched = dict()
        enriched["left_uri"] = p[0]
        enriched["right_uri"] = p[1]
        enriched["val"] = p[2]
        enriched["pred"] = p[3]
        left_types = type_dict[p[0]]
        right_types = type_dict[p[1]]

        # ScadsMB
        if superclasses is None:
            enriched["left_types"] = left_types
            enriched["right_types"] = right_types
        elif left_types == right_types:
            common = _find_best_general(left_types, superclasses)
            enriched["left_types"] = common
            enriched["right_types"] = common
        else:
            enriched["left_types"] = _find_best_general(left_types, superclasses)
            enriched["right_types"] = _find_best_general(right_types, superclasses)
        typed_preds.append(enriched)
    return typed_preds


def get_entity_node_degrees(ds_path: str) -> dict:
    entity_degree_dict = dict()
    for rel_file in glob.iglob(ds_path + "/rel_triples_*"):
        with open(rel_file, "r") as in_file:
            for line in in_file:
                triples = line.strip().split("\t")
                if not triples[0] in entity_degree_dict:
                    entity_degree_dict[triples[0]] = 0
                if not triples[2] in entity_degree_dict:
                    entity_degree_dict[triples[2]] = 0
                entity_degree_dict[triples[0]] += 1
                entity_degree_dict[triples[2]] += 1
    for att_file in glob.iglob(ds_path + "/attr_triples_*"):
        with open(att_file, "r") as in_file:
            for line in in_file:
                triples = line.strip().split("\t")
                if not triples[0] in entity_degree_dict:
                    entity_degree_dict[triples[0]] = 0
                entity_degree_dict[triples[0]] += 1
    # TODO check if this is correct?
    with open(ds_path + "/ent_links") as ent_file:
        for line in ent_file:
            duples = line.strip().split("\t")
            if not duples[0] in entity_degree_dict:
                entity_degree_dict[duples[0]] = 0
            if not duples[1] in entity_degree_dict:
                entity_degree_dict[duples[1]] = 0
            entity_degree_dict[duples[0]] += 1
            entity_degree_dict[duples[1]] += 1
    return entity_degree_dict


def average_node_degree(typed_pred: pd.DataFrame, entity_degrees: dict) -> pd.DataFrame:
    ed = pd.Series(entity_degrees).to_frame(name="node degree")
    ed.reset_index(inplace=True)
    ed_left = ed.rename(columns={"index": "left_uri"})
    ed_right = ed.rename(columns={"index": "right_uri"})
    left = ed_left.merge(typed_pred, on="left_uri")[
        ["left_uri", "left_types", "node degree"]
    ]
    right = ed_right.merge(typed_pred, on="right_uri")[
        ["right_uri", "right_types", "node degree"]
    ]
    left = left.drop_duplicates()
    right = right.drop_duplicates()
    left = left.rename(columns={"left_uri": "uri", "left_types": "Type"})
    right = right.rename(columns={"right_uri": "uri", "right_types": "Type"})
    return left.append(right).groupby("Type").mean()


def _pkl_path(pred_path: str, folder="/tmp/"):
    arr = pred_path.split("/")[-1].split("-")
    return (
        folder + arr[0] + arr[3].split(".")[0].replace("_test_pred", "typed") + ".pkl"
    )


def create_typed_predictions(
    ent_id_path1: List[str],
    ent_id_path2: List[str],
    pred_path: List[str],
    type_path,
    superclasses_path: str,
    cache=True,
) -> pd.DataFrame:
    """
    Create a dataframe with the columns: left_types, left_uri, pred, right_types, right_uri, val
    Which is the uri of the entity pair, the determined type, true value of match and predicted value
    @param ent_id_path1: paths of kg1 entity id files,
    @param ent_id_path2: paths of kg2 entity id files,
    @param pred_path: paths of predictions files,
    @param type_path: path(s) of type dict,
    @param superclasses_path: path of superclasses dict used to find more general types,
    @param cache: cache result,
    """
    pkl_path = _pkl_path(pred_path[0])
    if cache and os.path.exists(pkl_path):
        print(f"Load cached {pkl_path}")
        return pd.read_pickle(pkl_path)
    superclasses = None
    # typing is different for ScaDS datasets
    if "ScadsMB" not in superclasses_path:
        assert (
            len(ent_id_path1) == len(ent_id_path2) == len(pred_path) == len(type_path)
        )
        with open(superclasses_path, "r") as fp:
            superclasses = json.load(fp)

    overall_typed = []

    for i in range(len(ent_id_path1)):
        id_url_dicts = (
            get_id_url_dict(ent_id_path1[i]),
            get_id_url_dict(ent_id_path2[i]),
        )
        pred = read_pred(pred_path[i])
        pred = pred_with_url(pred, id_url_dicts)
        if "ScadsMB" not in superclasses_path:
            type_dict = read_json(type_path[i])
        else:
            type_dict = read_json(type_path)
        overall_typed = overall_typed + find_fitting_types(
            pred, type_dict, superclasses
        )
    df = pd.DataFrame(overall_typed)
    if cache:
        pd.to_pickle(df, pkl_path)
    return df


def create_combined_df(typed_pred: pd.DataFrame, avg_nd: pd.DataFrame,) -> pd.DataFrame:
    """
    Create a dataframe, that contains error rates and average node degrees per type
    @param typed_pred: df that was result of `create_typed_predictions`
    @param avg_nd df that was result of `average_node_degree`
    """
    total_fp_col_name = "fp"
    total_fn_col_name = "fn"
    total_all_col_name = "occurence"
    fp_col_name = "fp rate"
    fn_col_name = "fn rate"
    all_col_name = "rate of all"
    ###
    # get false positive and false negative rate per type
    ###
    false_negatives = typed_pred.loc[
        (typed_pred["pred"] == 0) & (typed_pred["val"] == 1)
    ]
    false_positives = typed_pred.loc[
        (typed_pred["pred"] == 1) & (typed_pred["val"] == 0)
    ]

    fp_type_counts = (
        false_positives["left_types"]
        .value_counts()
        .to_frame()
        .join(false_positives["right_types"].value_counts().to_frame())
    )
    fp_type_counts["types"] = (
        fp_type_counts["left_types"] + fp_type_counts["right_types"]
    )
    fp_type_counts.drop(labels=["left_types", "right_types"], axis=1, inplace=True)
    fn_type_counts = false_negatives["left_types"].value_counts().to_frame()

    tmp = (
        (
            typed_pred["left_types"].value_counts()
            + typed_pred["right_types"].value_counts()
        )
        / 2
    ).to_frame(name="types")
    tmp = tmp.join(fn_type_counts)
    tmp = tmp.join(fp_type_counts, lsuffix="ALL", rsuffix="FP")

    ###
    #  combine with avg nd
    ###
    combined = tmp.join(avg_nd)
    combined.reset_index(inplace=True)
    combined = combined.rename(
        columns={
            "index": "Type",
            "typesALL": total_all_col_name,
            "left_types": total_fn_col_name,
            "typesFP": total_fp_col_name,
            "node degree": "avg node degree",
        }
    )
    # get shorter type names
    combined["Type"] = [
        x.split("/")[-1].split("#")[-1] for x in combined["Type"].astype(str)
    ]
    # get percentages
    combined[fp_col_name] = (
        combined[total_fp_col_name] / combined[total_all_col_name]
    ) * 100
    combined[fn_col_name] = (
        combined[total_fn_col_name] / combined[total_all_col_name]
    ) * 100
    combined[all_col_name] = (
        combined[total_all_col_name] / combined[total_all_col_name].sum()
    ) * 100
    return combined.sort_values(by=all_col_name, ascending=False)


def create_scatter(
    combined: pd.DataFrame,
    x_axis_column: str,
    y_axis_column: str,
    x_axis_label: str,
    y_axis_label: str,
    file_path: str,
    multiple: bool,
):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if multiple:
        g = sns.scatterplot(
            x=x_axis_column, y=y_axis_column, hue=EA_col, style=EA_col, data=combined
        )
    else:
        g = sns.scatterplot(x=x_axis_column, y=y_axis_column, data=combined)
    for label in g.get_xticklabels():
        label.set_rotation(90)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()


def create_heatmap(data: pd.DataFrame, method: str, file_path: str):
    sns.heatmap(data.corr(method=method), annot=True)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()


def _get_files(
    embedding_approach: str,
    dataset_name: str,
    base_folder: str,
    vector_type="SimAndEmb",
):
    kg1_ent_id_files = sorted(
        [
            i
            for i in glob.iglob(
                f"{base_folder}/Embeddings/{embedding_approach}/{dataset_name}/721_5fold/*/*/kg1_ent_ids"
            )
        ]
    )
    kg2_ent_id_files = sorted(
        [
            i
            for i in glob.iglob(
                f"{base_folder}/Embeddings/{embedding_approach}/{dataset_name}/721_5fold/*/*/kg2_ent_ids"
            )
        ]
    )
    pred_files = sorted(
        [
            i
            for i in glob.iglob(
                f"{base_folder}/output/results/{dataset_name}-721_5fold-*/{embedding_approach}/datasets/*/{dataset_name}-721_5fold-*_random forest 500_{vector_type}_test_pred.csv"
            )
        ]
    )
    return kg1_ent_id_files, kg2_ent_id_files, pred_files


def create_melted_node_degree_frame(
    entity_degrees: dict, typed_pred: pd.DataFrame, left_name: str, right_name: str
) -> pd.DataFrame:
    """
    Creates df that is used for scatterplot of match type to node degree
    @param entity_degrees: dictionary with degrees of entities
    @param typed_pred: df that is result of `create_typed_predictions`
    @param left_name: name of left ds
    @param right_name: name of left ds

    @returns pd.DataFrame with columns: left_uri, right_uri, left_types, right_types, match_type, node degree, degree
    where node is used for legend
    """
    ed = pd.Series(entity_degrees).to_frame(name="node degree")
    ed.reset_index(inplace=True)
    ed_left = ed.rename(columns={"index": "left_uri"})
    ed_right = ed.rename(columns={"index": "right_uri"})
    left = ed_left.merge(typed_pred, on="left_uri")[["left_uri", "node degree"]]
    right = ed_right.merge(typed_pred, on="right_uri")[["right_uri", "node degree"]]
    left = left.drop_duplicates()
    right = right.drop_duplicates()
    left = left.rename(columns={"node degree": "left node degree"})
    right = right.rename(columns={"node degree": "right node degree"})
    df_nd = typed_pred.merge(left, on="left_uri").merge(right, on="right_uri")
    df_nd.loc[
        (df_nd["pred"] == 0) & (df_nd["val"] == 1), "match_type"
    ] = "False\nNegative"
    df_nd.loc[
        (df_nd["pred"] == 1) & (df_nd["val"] == 0), "match_type"
    ] = "False\nPositive"
    df_nd.loc[
        (df_nd["pred"] == 1) & (df_nd["val"] == 1), "match_type"
    ] = "True\nPositive"
    df_nd.loc[
        (df_nd["pred"] == 0) & (df_nd["val"] == 0), "match_type"
    ] = "True\nNegative"
    df_nd = df_nd.rename(
        columns={"left node degree": left_name, "right node degree": right_name}
    )
    return df_nd.melt(
        id_vars=["left_uri", "right_uri", "left_types", "right_types", "match_type"],
        value_vars=[left_name, right_name],
        var_name="node degree",
        value_name="degree",
    )


def _get_ds_names(dataset_name: str):
    sep = "_"
    if sep not in dataset_name:
        sep = "-"
    left_name = dataset_name.split(sep)[0]
    right_name = dataset_name.split(sep)[1]
    return ds_name_long[left_name], ds_name_long[right_name]


def create_combined_over_embeddings(
    embedding_approaches: List[str],
    dataset_name: str,
    type_files: List[str],
    base_folder: str,
    data_source=None,
    vector_type="SimAndEmb",
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    @param embedding_approaches: list of embedding approaches names
    @param dataset_name: name of current ds
    @param type_files: paths of type files
    @param base_folder: folder path where data is stored
    @param data_source: OpenEA or ScaDS path, if None will try to guess by dataset_name
    @param vector_type: type of input e.g. "SimAndEmb"

    @returns list of typed df and list of melted dfs to be used for scatterplot of match type and node degree
    """

    if data_source is None:
        if "15K" in dataset_name or "100K" in dataset_name:
            data_source = "OpenEA"
        else:
            data_source = "EA-ScaDS-Datasets/ScadsMB"

    dfs = []
    melted = []
    entity_degrees = get_entity_node_degrees(
        f"{base_folder}/{data_source}/{dataset_name}"
    )
    for e in tqdm(embedding_approaches, desc="Create combined df"):
        kg1_ent_id_files, kg2_ent_id_files, pred_files = _get_files(
            e, dataset_name, base_folder, vector_type
        )
        df = create_typed_predictions(
            kg1_ent_id_files,
            kg2_ent_id_files,
            pred_files,
            type_files,
            f"{base_folder}/{data_source}/typed_links/superclasses.json",
        )
        left_name, right_name = _get_ds_names(dataset_name)
        melted.append(
            create_melted_node_degree_frame(entity_degrees, df, left_name, right_name)
        )
        avg_nd = average_node_degree(df, entity_degrees)
        combined_inner = create_combined_df(df, avg_nd)
        combined_inner[EA_col] = e
        dfs.append(combined_inner)
    return dfs[0].append(dfs[1]).append(dfs[2]), melted


def create_stripplot(data: pd.DataFrame, emb_approach_name: str, output_folder: str):
    with sns.axes_style("darkgrid", {"ytick.left": True}):
        g = sns.stripplot(
            x="match_type",
            y="degree",
            hue="node degree",
            dodge=True,
            jitter=0.35,
            size=2,
            data=data,
        )
        g.set_yscale("log")
        plt.savefig(
            f"{output_folder}/{emb_approach_name}_stripplot.png", bbox_inches="tight"
        )
        plt.close()


if __name__ == "__main__":
    embedding_approaches = ["BootEA", "MultiKE", "RDGCN"]
    dataset_name = sys.argv[1]
    data_source = "OpenEA"
    if len(sys.argv) == 3:
        data_source = "EA-ScaDS-Datasets/ScadsMB"

    output_folder = f"output/figures/{dataset_name}"
    graph_args = [
        (
            "Type",
            "fn rate",
            "Type",
            "False negative rate in %",
            f"{output_folder}/type_fn.png",
            True,
        ),
        (
            "Type",
            "fp rate",
            "Type",
            "False positive rate in %",
            f"{output_folder}/type_fp.png",
            True,
        ),
        (
            "Type",
            "rate of all",
            "Type",
            "Percent of all",
            f"{output_folder}/type_off_all.png",
            False,
        ),
        (
            "Type",
            "avg node degree",
            "Type",
            "Average node degree",
            f"{output_folder}/type_node_degree.png",
            False,
        ),
        (
            "avg node degree",
            "fn rate",
            "Average node degree",
            "False negative rate in %",
            f"{output_folder}/avg_nd_fn.png",
            True,
        ),
        (
            "avg node degree",
            "fp rate",
            "Average node degree",
            "False positive rate in %",
            f"{output_folder}/avg_nd_fp.png",
            True,
        ),
        (
            "rate of all",
            "fn rate",
            "Percent of all",
            "False negative rate in %",
            f"{output_folder}/rate_of_all_fn.png",
            True,
        ),
        (
            "rate of all",
            "fp rate",
            "Percent of all",
            "False positive rate in %",
            f"{output_folder}/rate_of_all_fp.png",
            True,
        ),
    ]
    if "ScaDS" in data_source:
        type_files = f"/home/dobraczka/Downloads/git/er-embedding-benchmark/data/{data_source}/typed_links/datasets/{dataset_name}"
    else:
        type_files = sorted(
            [
                i
                for i in glob.iglob(
                    f"/home/dobraczka/Downloads/git/er-embedding-benchmark/data/{data_source}/typed_links/datasets/{dataset_name}/721_5fold/*/typed_test"
                )
            ]
        )
    combined, melted = create_combined_over_embeddings(
        embedding_approaches,
        dataset_name,
        type_files,
        "/home/dobraczka/Downloads/git/er-embedding-benchmark/data/",
        data_source,
    )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for e, m in zip(embedding_approaches, melted):
        create_stripplot(m, e, output_folder)

    for args in graph_args:
        create_scatter(combined, *args)
    create_heatmap(
        combined[["fp rate", "fn rate", "avg node degree", "rate of all"]],
        "spearman",
        f"{output_folder}/spearman_corr.png",
    )
    create_heatmap(
        combined[["fp rate", "fn rate", "avg node degree", "rate of all"]],
        "kendall",
        f"{output_folder}/kendall_corr.png",
    )
