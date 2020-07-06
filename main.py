import json
import os
import time
from glob import glob
from os import path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from openea.models.basic_model import BasicModel

from dataset.dataset import Dataset
from matching.matcher import MatchModelTrainer
from matching.pair_to_vec import (
    SimAndEmb,
    SimAndEmbNormalized,
    OnlySim,
    OnlySimNormalized,
    OnlyEmb,
    PairToVec,
)
from prepared_models import models
from run_configs import configs
from similarity.create_training import create_feature_similarity_frame
from utils.file_printer import FileAndConsole


def main():
    for i in range(len(configs)):
        run_for_dataset(i)


def run_for_dataset(dataset_idx):
    import tensorflow as tf

    tf.reset_default_graph()
    dataset, (embedding_model, embedding_name, embedding_fn) = configs[dataset_idx]()
    dataset.add_negative_samples()
    print(f"using {embedding_name} on {dataset.name()}")

    kgs = dataset.kgs()
    embedding_model.args.output = f"output/results/{dataset.name().replace('/', '-')}/"
    embedding_model.set_args(embedding_model.args)

    existing_folder = find_existing_result_folder(embedding_model)
    existing_embedding_folder = find_existing_embedding_folder(embedding_model, dataset)

    output_folder = existing_folder or embedding_model.out_folder[:-1]
    progress_file = (
        f"{output_folder}/{dataset.name().replace('/', '-')}-{embedding_name}.txt"
    )
    csv_result_file = progress_file.replace(".txt", ".csv")
    if path.exists(progress_file) and path.exists(csv_result_file):
        return

    if existing_embedding_folder is not None:
        embeddings = np.load(f"{existing_embedding_folder}/ent_embeds.npy")
    else:
        embedding_model.set_kgs(kgs)
        embedding_model.init()
        embedding_model.run()
        embedding_model.test()
        embedding_model.save()
        embeddings = embedding_fn(embedding_model)

    all_pairs = (
        dataset.labelled_train_pairs
        + dataset.labelled_val_pairs
        + dataset.labelled_test_pairs
    )
    sims_file = f"{existing_folder}/all_sims.parquet"
    if existing_folder is not None and path.exists(sims_file):
        all_sims = pd.read_parquet(sims_file)
        print("loaded all_sims from file")
        min_max = joblib.load(f"{existing_folder}/min_max.pkl")
        print("loaded min_max parameters from file")
        with open(f"{existing_folder}/scale_cols.json") as f:
            scale_cols = json.load(f)
    else:
        all_sims, min_max, scale_cols = create_feature_similarity_frame(
            embeddings, all_pairs, kgs, only_training=True,
        )
        output_folder = existing_folder or embedding_model.out_folder[:-1]
        if not path.exists(output_folder):
            dir_path = output_folder.split("/")
            for i in range(len(dir_path)):
                if not path.exists("/".join(dir_path[: i + 1])):
                    os.mkdir("/".join(dir_path[: i + 1]))
        all_sims.to_parquet(f"{output_folder}/all_sims.parquet")
        joblib.dump(min_max, f"{output_folder}/min_max.pkl")
        with open(f"{output_folder}/scale_cols.json", "w") as f:
            json.dump(scale_cols, f)
    all_sims = all_sims.dropna(axis=1, how="all", thresh=int(0.1 * len(all_sims)))

    pair_to_vecs = [
        SimAndEmb(embeddings, all_sims, min_max, scale_cols, kgs),
        SimAndEmbNormalized(embeddings, all_sims, min_max, scale_cols, kgs),
        OnlySim(embeddings, all_sims, min_max, scale_cols, kgs),
        OnlySimNormalized(embeddings, all_sims, min_max, scale_cols, kgs),
        OnlyEmb(embeddings, all_sims, min_max, scale_cols, kgs),
    ]

    file_to_print = FileAndConsole(progress_file)
    results_list = []
    for pair_to_vec in pair_to_vecs:
        for model in models:
            model = model(pair_to_vec)
            print(f"\n{model} - {type(pair_to_vec).__name__}", file=file_to_print)
            results_list.append(run(model, dataset, pair_to_vec, file_to_print))
    file_to_print.close()
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
    results.to_csv(csv_result_file, index=False)


def find_existing_result_folder(embedding_model: BasicModel):
    folder = embedding_model.out_folder
    folder = "/".join(folder.split("/")[:-2])

    dirs = sorted(
        [d for d in glob(f"{folder}/*") if embedding_model.args.dataset_division in d]
    )
    if any(dirs):
        return dirs[-1]
    return None


def find_existing_embedding_folder(embedding_model: BasicModel, dataset: Dataset):
    folder = f"data/Embeddings15K/{type(embedding_model).__name__}/{dataset.name()}"
    # we're using a OpenEA dataset
    if not path.exists(folder):
        folder = embedding_model.out_folder
        folder = "/".join(folder.split("/")[:-2])
        dirs = sorted(
            [
                d
                for d in glob(f"{folder}/*")
                if embedding_model.args.dataset_division in d
            ]
        )
    else:
        dirs = sorted([d for d in glob(f"{folder}/*")])

    if any(dirs):
        return dirs[-1]
    return None


def run(
    model_trainer: MatchModelTrainer, dataset: Dataset, pair_to_vec: PairToVec, file
) -> Dict[str, Any]:
    start = time.time()
    model_trainer.fit(dataset.labelled_train_pairs, dataset.labelled_val_pairs)
    train_time = time.time() - start
    train_eval = model_trainer.evaluate(dataset.labelled_train_pairs)
    print(f"train: {train_eval}", file=file)
    valid_eval = model_trainer.evaluate(dataset.labelled_val_pairs)
    print(f"valid: {valid_eval}", file=file)
    test_start = time.time()
    test_eval = model_trainer.evaluate(dataset.labelled_test_pairs)
    test_time = time.time() - test_start
    print(f"test: {test_eval}", file=file)
    return {
        "model_name": model_trainer.__str__(),
        "vector_name": type(pair_to_vec).__name__,
        "train_precision": train_eval.precision,
        "train_recall": train_eval.recall,
        "train_f1": train_eval.f1,
        "val_precision": valid_eval.precision,
        "val_recall": valid_eval.recall,
        "val_f1": valid_eval.f1,
        "test_precision": test_eval.precision,
        "test_recall": test_eval.recall,
        "test_f1": test_eval.f1,
        "train_time": train_time,
        "test_time": test_time,
    }


if __name__ == "__main__":
    main()
