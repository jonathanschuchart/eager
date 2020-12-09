from glob import glob
from typing import Any, Dict, List
from os import path, mkdir

import pandas as pd
from openea.models.basic_model import BasicModel


def write_result_files(
    output_folder, dataset, embedding_name, results_list: List[Dict[str, Any]]
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

    test_pred = [
        (
            "test",
            r["model_name"],
            r["vector_name"],
            [
                [*a, b]
                for a, b in zip(
                    r["test_prediction"], [e[2] for e in dataset.labelled_test_pairs]
                )
            ],
        )
        for r in results_list
    ]
    train_pred = [
        (
            "train",
            r["model_name"],
            r["vector_name"],
            [
                [*a, b]
                for a, b in zip(
                    r["train_prediction"], [e[2] for e in dataset.labelled_train_pairs]
                )
            ],
        )
        for r in results_list
    ]
    val_pred = [
        (
            "val",
            r["model_name"],
            r["vector_name"],
            [
                [*a, b]
                for a, b in zip(
                    r["val_prediction"], [e[2] for e in dataset.labelled_val_pairs]
                )
            ],
        )
        for r in results_list
    ]
    for typ, model, vec, pred in test_pred + train_pred + val_pred:
        pd.DataFrame(data=pred, columns=["left", "right", "pred", "val"]).to_csv(
            csv_file.replace(".csv", f"_{model}_{vec}_{typ}_pred.csv"), index=False,
        )

    if path.exists(csv_file):
        old_frame = pd.read_csv(csv_file)
        results = merge_dataframes(old_frame, results)
    results.to_csv(csv_file, index=False)


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
