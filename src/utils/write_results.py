from glob import glob
from typing import Any, Dict, List
from os import path, mkdir

import pandas as pd
from openea.models.basic_model import BasicModel

from dataset.dataset import Dataset


def write_result_file(
    output_folder: str,
    dataset: Dataset,
    embedding_name: str,
    results_list: List[Dict[str, Any]],
):
    _ensure_folder_exists(output_folder)
    csv_file = (
        f"{output_folder}/{dataset.name().replace('/', '-')}-{embedding_name}.csv"
    )
    results = pd.DataFrame(
        data=results_list,
        columns=[
            "model_name",
            "vector_name",
            "train_precision",
            "train_recall",
            "train_f1",
            "val_precision",
            "val_recall",
            "val_f1",
            "test_precision",
            "test_recall",
            "test_f1",
            "train_time",
            "test_time",
        ],
    )

    # don't overwrite old results from different parameters
    if path.exists(csv_file):
        old_frame = pd.read_csv(csv_file)
        results = merge_dataframes(old_frame, results)
    results.to_csv(csv_file, index=False)
    print(f"Written results file {csv_file}")


def write_predictions(output_folder, artifacts, dataset, embinfo, classifier, vec):
    _ensure_folder_exists(output_folder)
    train, val, test = [
        [[*a, b] for a, b in zip(pred, [e[2] for e in dataset.labelled_val_pairs])]
        for pred, labels in [
            (artifacts["train_prediction"], dataset.labelled_train_pairs),
            (artifacts["val_prediction"], dataset.labelled_val_pairs),
            (artifacts["test_prediction"], dataset.labelled_test_pairs),
        ]
    ]

    f_names = []
    for pred, typ in [(train, "train"), (val, "val"), (test, "test")]:
        file_name = path.join(
            output_folder,
            f"{dataset.name().replace('/', '-')}_{embinfo.name}_{classifier.replace(' ', '-')}_{vec}_{typ}_pred.csv",
        )
        pd.DataFrame(data=pred, columns=["left", "right", "pred", "val"]).to_csv(
            file_name, index=False,
        )
        f_names.append(file_name)
    return f_names


def _ensure_folder_exists(output_folder):
    if not path.exists(output_folder):
        dir_path = output_folder.split("/")
        for i in range(len(dir_path)):
            if not path.exists("/".join(dir_path[: i + 1])):
                mkdir("/".join(dir_path[: i + 1]))


def merge_dataframes(old_frame: pd.DataFrame, new_frame):
    old_frame.set_index(keys=["model_name", "vector_name"], drop=False, inplace=True)
    new_frame.set_index(keys=["model_name", "vector_name"], drop=False, inplace=True)

    return new_frame.combine_first(old_frame)


def find_existing_result_folder(embedding_model: BasicModel):
    folder = embedding_model.out_folder
    folder = "/".join(folder.split("/")[:-2])

    dirs = sorted(
        [d for d in glob(f"{folder}/*") if embedding_model.args.dataset_division in d]
    )
    if any(dirs):
        return dirs[-1]
    return None
