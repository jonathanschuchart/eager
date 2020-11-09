import json
import os
import random
import time
from glob import glob
from multiprocessing import Pool
from os import path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from openea.models.basic_model import BasicModel

from attribute_features import CartesianCombination
from dataset.dataset import Dataset
from distance_measures import NumberDistance, DateDistance, EmbeddingEuclideanDistance
from experiment import Experiments, Experiment
from matching.matcher import MatchModelTrainer
from matching.pair_to_vec import PairToVec
from prepared_models import model_factories
from run_configs import configs, init_configs
from similarity.create_training import create_feature_similarity_frame
from similarity.measure_finding import init_measures
from similarity_measures import (
    Levenshtein,
    GeneralizedJaccard,
    TriGram,
    EmbeddingConcatenation, NumberSimilarity,
)


def main():
    for i in range(len(configs)):
        run_for_dataset(i)


def run_for_dataset(dataset_idx):
    import tensorflow as tf

    tf.reset_default_graph()

    rnd = random.Random(42)
    dataset, (embedding_model, embedding_name, embedding_fn) = configs[dataset_idx]()
    dataset.add_negative_samples(rnd)
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

    # start = time.time()
    all_sims, min_max, scale_cols = create_feature_similarity_frame(
        embeddings, all_pairs, kgs, only_training=True,
    )
    output_folder = existing_folder or embedding_model.out_folder[:-1]
    # if not path.exists(output_folder):
    #     dir_path = output_folder.split("/")
    #     for i in range(len(dir_path)):
    #         if not path.exists("/".join(dir_path[: i + 1])):
    #             os.mkdir("/".join(dir_path[: i + 1]))
    # all_sims.to_parquet(f"{output_folder}/all_sims.parquet")
    # joblib.dump(min_max, f"{output_folder}/min_max.pkl")
    # with open(f"{output_folder}/scale_cols.json", "w") as f:
    #     json.dump(scale_cols, f)
    #
    print(f"original len: {len(all_sims)}")
    print(f"original before: {len(all_sims.columns)}")
    all_sims = all_sims.dropna(axis=1, how="all", thresh=int(0.1 * len(all_sims)))
    print(f"original after: {len(all_sims.columns)}")
    print(f"original: {list(all_sims.columns)}")
    sim = all_sims.loc[(all_pairs[0][0], all_pairs[0][1])]
    print([(k, sim.get(k, 0.0)) for k in all_sims.columns])
    # with open(csv_result_file.replace(".csv", "_df_times.csv"), "w") as f:
    #     f.write(f"{dataset.name()},{embedding_name},{time.time() - start}")

    # pair_to_vecs = [
    #     SimAndEmb(embeddings, all_sims, min_max, scale_cols, kgs),
    #     OnlySim(embeddings, all_sims, min_max, scale_cols, kgs),
    #     OnlyEmb(embeddings, all_sims, min_max, scale_cols, kgs),
    # ]

    # run_params = [
    #     (model_fac(pair_to_vec), dataset, pair_to_vec)
    #     for pair_to_vec in pair_to_vecs
    #     for model_fac in model_factories
    # ]

    cartesian_attr_combination = CartesianCombination(
        kgs,
        [NumberSimilarity()],
        [DateDistance()],
        [Levenshtein(), GeneralizedJaccard(), TriGram()],
    )
    no_attribute_combinations = CartesianCombination(kgs, [], [], [])
    pair_to_vecs = [
        # PairToVec(
        #     embeddings,
        #     kgs,
        #     "SimAndEmb",
        #     cartesian_attr_combination,
        #     [EmbeddingEuclideanDistance(), EmbeddingConcatenation()],
        # ),
        # PairToVec(
        #     embeddings,
        #     kgs,
        #     "OnlyEmb",
        #     no_attribute_combinations,
        #     [EmbeddingConcatenation()],
        # ),
        PairToVec(embeddings, kgs, "OnlySim", cartesian_attr_combination, []),
    ]
    for pvp in pair_to_vecs:
        pvp.prepare(all_pairs)

    print([(k, v) for k, v in zip(pair_to_vecs[-1].all_keys, pair_to_vecs[-1](all_pairs[0][0], all_pairs[0][1]))])
    experiments = Experiments(
        output_folder,
        [
            Experiment(model_fac)
            for pair_to_vec in pair_to_vecs
            for model_fac in model_factories(pair_to_vec)
        ],
        dataset,
    )
    results_list = experiments.run()

    # with Pool(processes=4) as pool:
    #     results_list = pool.starmap(run, run_params)

    # results_list = [run(model, dataset, pair_to_vec) for model, dataset, pair_to_vec in run_params]

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
        ("test", r["model_name"], r["vector_name"], r["test_prediction"])
        for r in results_list
    ]
    train_pred = [
        ("train", r["model_name"], r["vector_name"], r["train_prediction"])
        for r in results_list
    ]
    val_pred = [
        ("val", r["model_name"], r["vector_name"], r["val_prediction"])
        for r in results_list
    ]
    for typ, model, vec, pred in test_pred + train_pred + val_pred:
        pd.DataFrame(data=pred, columns=["left", "right", "val", "pred"]).to_csv(
            csv_result_file.replace(".csv", f"_{model}_{vec}_{typ}_pred.csv"),
            index=False,
        )

    if path.exists(csv_result_file):
        old_frame = pd.read_csv(csv_result_file)
        results = merge_dataframes(old_frame, results)
    results.to_csv(csv_result_file, index=False)


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


def find_existing_embedding_folder(embedding_model: BasicModel, dataset: Dataset):
    folder = f"data/Embeddings{dataset.data_size.value}K/{type(embedding_model).__name__}/{dataset.name()}"
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
    model_trainer: MatchModelTrainer, dataset: Dataset, pair_to_vec: PairToVec
) -> Dict[str, Any]:
    print(f"\n{model_trainer} - {type(pair_to_vec).__name__}")
    start = time.time()
    model_trainer.fit(dataset.labelled_train_pairs, dataset.labelled_val_pairs)
    train_time = time.time() - start
    train_eval = model_trainer.evaluate(dataset.labelled_train_pairs)
    print(f"train: {train_eval}")
    valid_eval = model_trainer.evaluate(dataset.labelled_val_pairs)
    print(f"valid: {valid_eval}")
    test_start = time.time()
    test_eval = model_trainer.evaluate(dataset.labelled_test_pairs)
    test_time = time.time() - test_start
    print(f"test: {test_eval}")
    return {
        "model_name": model_trainer.__str__(),
        "vector_name": type(pair_to_vec).__name__,
        "train_precision": train_eval.precision,
        "train_recall": train_eval.recall,
        "train_f1": train_eval.f1,
        "train_prediction": train_eval.prediction,
        "val_precision": valid_eval.precision,
        "val_recall": valid_eval.recall,
        "val_f1": valid_eval.f1,
        "val_prediction": valid_eval.prediction,
        "test_precision": test_eval.precision,
        "test_recall": test_eval.recall,
        "test_f1": test_eval.f1,
        "test_prediction": test_eval.prediction,
        "train_time": train_time,
        "test_time": test_time,
    }


if __name__ == "__main__":
    import torch

    torch.multiprocessing.set_start_method("spawn")
    init_measures()
    init_configs()
    main()
