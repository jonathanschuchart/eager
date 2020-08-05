import glob
import json
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm

too_broad = {
    "http://www.w3.org/2002/07/owl#Thing",
    "http://dbpedia.org/class/yago/Object100002684",
    "http://dbpedia.org/class/yago/PhysicalEntity100001930",
    "http://dbpedia.org/class/yago/Whole100003553",
    "http://dbpedia.org/ontology/Work",
    "http://www.wikidata.org/entity/Q386724",
    "http://schema.org/CreativeWork",
    "http://dbpedia.org/class/yago/YagoPermanentlyLocatedEntity",
    "http://dbpedia.org/class/yago/Abstraction100002137",
    "http://dbpedia.org/class/yago/Artifact100021939",
}


def get_type_occurences(dataset_path: str) -> dict:
    """
    Creates dictionary with occurence of types for dataset
    @param dataset_path:  path of dataset containing all json files with type dictionaries
    @return:  dictionaray with types as keys and number of occurences in this dataset
    """
    occurences = dict()
    for type_dict_file in tqdm(
        glob.iglob(dataset_path + "/721_5fold/*/typed_*"),
        desc="Getting type occurences",
    ):
        with open(type_dict_file) as fp:
            type_dict = json.load(fp)
            for _, type_list in type_dict.items():
                for t in type_list:
                    if "dbpedia" in t and "yago" not in t and "wikidata" not in t:
                        if t not in too_broad:
                            if t not in occurences:
                                occurences[t] = 1
                            occurences[t] += 1
    return occurences


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


def find_fitting_types(
    pred: List[List], type_dict: dict, type_occurences: dict, most_common: int
):
    typed_preds = []
    for p in tqdm(pred, desc="Enrich predictions"):
        enriched = dict()
        enriched["left_uri"] = p[0]
        enriched["right_uri"] = p[1]
        enriched["val"] = p[2]
        enriched["pred"] = p[3]
        left_types = type_dict[p[0]]
        right_types = type_dict[p[1]]
        if left_types == right_types:
            common = _find_most_common(left_types, type_occurences, most_common)
            enriched["left_types"] = common
            enriched["right_types"] = common
        else:
            enriched["left_types"] = _find_most_common(
                left_types, type_occurences, most_common
            )
            enriched["right_types"] = _find_most_common(
                right_types, type_occurences, most_common
            )
        typed_preds.append(enriched)
    return typed_preds


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
    type_occ = get_type_occurences(type_dataset_path)
    type_occ = sorted(type_occ.items(), key=lambda x: x[1], reverse=True)

    overall_typed = []

    for i in range(len(ent_id_path1)):
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
