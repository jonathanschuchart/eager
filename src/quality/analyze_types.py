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
EA_col = "Embedding approach"


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


def pred_with_url(pred: List[List], dicts: Tuple[dict, dict], only_wrong) -> List[List]:
    """
    Exchange ids in predictions with urls
    @param pred: predictions
    @param dicts: tuple of id->url dicts, tuple has to be in same order as entities in pred
    @param only_wrong: if True don't return correct predictions
    """
    urled_pred = []
    for p in pred:
        enriched = []
        if only_wrong and p[2] == p[3]:
            continue
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


def _find_most_common(types: List[str], type_occurences: dict, most_common: int):
    filtered_types = []
    for occ in type_occurences:
        if occ[0] in types:
            filtered_types.append(occ[0])
        if len(filtered_types) == most_common:
            if most_common == 1:
                return occ[0]
            break
    if len(filtered_types) == 0:
        if "http://www.w3.org/2002/07/owl#Thing" in types:
            return "http://www.w3.org/2002/07/owl#Thing"
        else:
            return "UNKNOWN"
    return filtered_types


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
    candidates = set()
    for t in types:
        if t in wanted:
            return t
        if t in superclasses:
            class_hierarchy = superclasses[t].copy()
            class_hierarchy.reverse()
            if class_hierarchy is None:
                continue
            if len(class_hierarchy) == 0:
                candidates.add(t)
            else:
                if t not in class_hierarchy:
                    class_hierarchy.append([t])
                candidates.add(_find_best_type_from_multiple(class_hierarchy))
    if len(candidates) == 1:
        return next(iter(candidates))
    return _find_best_type_from_multiple(candidates)


def find_fitting_types(
    pred: List[List], type_dict: dict, type_occurences: dict, most_common: int
):
    # TODO rename type_occ
    typed_preds = []
    for p in pred:
        enriched = dict()
        enriched["left_uri"] = p[0]
        enriched["right_uri"] = p[1]
        enriched["val"] = p[2]
        enriched["pred"] = p[3]
        left_types = type_dict[p[0]]
        right_types = type_dict[p[1]]
        if left_types == right_types:
            # common = _find_most_common(left_types, type_occurences, most_common)
            common = _find_best_general(left_types, type_occurences)
            enriched["left_types"] = common
            enriched["right_types"] = common
        else:
            # enriched["left_types"] = _find_most_common(
            #     left_types, type_occurences, most_common
            # )
            # enriched["right_types"] = _find_most_common(
            #     right_types, type_occurences, most_common
            # )
            enriched["left_types"] = _find_best_general(left_types, type_occurences)
            enriched["right_types"] = _find_best_general(right_types, type_occurences)
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


def create_typed_predictions(
    ent_id_path1: List[str],
    ent_id_path2: List[str],
    pred_path: List[str],
    type_path: List[str],
    type_dataset_path: str,
    most_common=1,
    only_wrong=True,
):
    assert len(ent_id_path1) == len(ent_id_path2) == len(pred_path) == len(type_path)
    # type_occ = get_type_occurences(type_dataset_path)
    # type_occ = sorted(type_occ.items(), key=lambda x: x[1], reverse=True)
    with open(type_dataset_path, "r") as fp:
        type_occ = json.load(fp)

    overall_typed = []

    for i in tqdm(range(len(ent_id_path1)), desc="Create typed predictions"):
        id_url_dicts = (
            get_id_url_dict(ent_id_path1[i]),
            get_id_url_dict(ent_id_path2[i]),
        )
        pred = read_pred(pred_path[i])
        pred = pred_with_url(pred, id_url_dicts, only_wrong)
        type_dict = read_json(type_path[i])
        overall_typed = overall_typed + find_fitting_types(
            pred, type_dict, type_occ, most_common
        )
    return pd.DataFrame(overall_typed)


def create_combined_df(
    typed_pred: pd.DataFrame,
    avg_nd: pd.DataFrame,
    fp_col_name="fp rate",
    fn_col_name="fn rate",
    all_col_name="rate of all",
) -> pd.DataFrame:
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
            "typesALL": all_col_name,
            "left_types": fn_col_name,
            "typesFP": fp_col_name,
            "node degree": "avg node degree",
        }
    )
    # get shorter type names
    combined["Type"] = [
        x.split("/")[-1].split("#")[-1] for x in combined["Type"].astype(str)
    ]
    # get percentages
    combined[fp_col_name] = (combined[fp_col_name] / combined[all_col_name]) * 100
    combined[fn_col_name] = (combined[fn_col_name] / combined[all_col_name]) * 100
    combined[all_col_name] = (
        combined[all_col_name] / combined[all_col_name].sum()
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


def _get_files(embedding_approach: str, dataset_name: str, base_folder: str):
    kg1_ent_id_files = sorted(
        [
            i
            for i in glob.iglob(
                f"{base_folder}/Embeddings15K/{embedding_approach}/{dataset_name}/721_5fold/*/*/kg1_ent_ids"
            )
        ]
    )
    kg2_ent_id_files = sorted(
        [
            i
            for i in glob.iglob(
                f"{base_folder}/Embeddings15K/{embedding_approach}/{dataset_name}/721_5fold/*/*/kg2_ent_ids"
            )
        ]
    )
    pred_files = sorted(
        [
            i
            for i in glob.iglob(
                f"{base_folder}/output/results/{dataset_name}-721_5fold-*/{embedding_approach}/datasets/*/{dataset_name}-721_5fold-*_random forest 500_SimAndEmb_test_pred.csv"
            )
        ]
    )
    return kg1_ent_id_files, kg2_ent_id_files, pred_files


def create_combined_over_embeddings(
    embedding_approaches: List[str], dataset_name, type_files, base_folder="data"
):
    dfs = []
    entity_degrees = get_entity_node_degrees(f"{base_folder}/OpenEA/{dataset_name}")
    for e in tqdm(embedding_approaches, desc="Create combined df"):
        kg1_ent_id_files, kg2_ent_id_files, pred_files = _get_files(
            e, dataset_name, base_folder
        )
        df = create_typed_predictions(
            kg1_ent_id_files,
            kg2_ent_id_files,
            pred_files,
            type_files,
            f"{base_folder}/OpenEA/typed_links/superclasses.json",
            1,
            False,
        )
        avg_nd = average_node_degree(df, entity_degrees)
        combined_inner = create_combined_df(df, avg_nd)
        combined_inner[EA_col] = e
        dfs.append(combined_inner)
    return dfs[0].append(dfs[1]).append(dfs[2])


if __name__ == "__main__":
    embedding_approaches = ["BootEA", "MultiKE", "RDGCN"]
    dataset_name = sys.argv[1]
    graph_args = [
        (
            "Type",
            "fn rate",
            "Type",
            "False negative rate in percent",
            f"output/figures/{dataset_name}/type_fn.png",
            True,
        ),
        (
            "Type",
            "fp rate",
            "Type",
            "False positive rate in percent",
            f"output/figures/{dataset_name}/type_fp.png",
            True,
        ),
        (
            "Type",
            "rate of all",
            "Type",
            "Percent of all",
            f"output/figures/{dataset_name}/type_off_all.png",
            True,
        ),
        (
            "Type",
            "avg node degree",
            "Type",
            "Average node degree",
            f"output/figures/{dataset_name}/type_node_degree.png",
            True,
        ),
    ]
    type_files = sorted(
        [
            i
            for i in glob.iglob(
                f"data/OpenEA/typed_links/datasets/{dataset_name}/721_5fold/*/typed_test"
            )
        ]
    )
    combined = create_combined_over_embeddings(
        embedding_approaches, type_files, dataset_name
    )
    for args in graph_args:
        create_scatter(combined, *args)
