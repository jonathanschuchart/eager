import random
import sys
from glob import glob
from os import path
from typing import List

import numpy as np
from openea.models.basic_model import BasicModel

from attribute_features import CartesianCombination, AllToOneCombination
from dataset.dataset import Dataset, DataSize
from distance_measures import DateDistance, EmbeddingEuclideanDistance
from eager import Eager
from experiment import Experiment
from matching.pair_to_vec import PairToVec
from prepared_models import classifiers
from run_configs import EmbeddingInfo
from similarity_measures import (
    Levenshtein,
    GeneralizedJaccard,
    TriGram,
    EmbeddingConcatenation,
    NumberSimilarity,
    BertCosineSimilarity,
    BertConcatenation,
)
from utils.argument_parser import parse_arguments
from utils.bert import create_bert_embeddings
from utils.write_results import (
    find_existing_result_folder,
    write_result_file,
    write_predictions,
)
from run_configs import boot_ea, multi_ke, rdgcn, get_config

embedding_measures = [EmbeddingEuclideanDistance(), EmbeddingConcatenation()]
cartesian_attr_combination = CartesianCombination(
    [NumberSimilarity()],
    [DateDistance()],
    [Levenshtein(), GeneralizedJaccard(), TriGram()],
)
no_attribute_combinations = CartesianCombination([], [], [])

all_to_one_concat = AllToOneCombination(
    [Levenshtein(), GeneralizedJaccard(), TriGram()]
)


def pair_to_vec_config(kgs, embeddings, bert_emb_folder, name):
    return {
        "SimAndEmb": lambda: PairToVec(
            embeddings, kgs, "SimAndEmb", cartesian_attr_combination, embedding_measures
        ),
        "OnlySim": lambda: PairToVec(
            embeddings, kgs, "OnlySim", cartesian_attr_combination, []
        ),
        "SimConcatAndEmb": lambda: PairToVec(
            embeddings, kgs, "SimAndEmb", all_to_one_concat, embedding_measures
        ),
        "OnlySimConcat": lambda: PairToVec(
            embeddings, kgs, "OnlySimConcat", all_to_one_concat, []
        ),
        "OnlyEmb": lambda: PairToVec(
            embeddings, kgs, "OnlyEmb", no_attribute_combinations, embedding_measures
        ),
        "BertConcatAndEmb": lambda: PairToVec(
            embeddings,
            kgs,
            "BertConcatAndEmb",
            AllToOneCombination(
                [
                    BertConcatenation(bert_emb_folder),
                    BertCosineSimilarity(bert_emb_folder),
                ]
            ),
            embedding_measures,
        ),
        "OnlyBertConcat": lambda: PairToVec(
            embeddings,
            kgs,
            "BertConcatAndEmb",
            AllToOneCombination(
                [
                    BertConcatenation(bert_emb_folder),
                    BertCosineSimilarity(bert_emb_folder),
                ]
            ),
            [],
        ),
    }[name]()


def main(cl_args):
    args = parse_arguments(cl_args)
    run_all(
        args.emb_models,
        args.data_paths,
        args.folds,
        args.sizes,
        args.classifiers,
        args.ptv_names,
    )


def run_single(
    dataset: Dataset,
    emb_info: EmbeddingInfo,
    output_folder,
    classifier_name: str,
    ptv_name: str,
):
    import tensorflow as tf

    tf.reset_default_graph()

    rnd = random.Random(42)
    dataset.add_negative_samples(rnd)
    embeddings = get_embeddings(dataset, emb_info)

    ptv = pair_to_vec_config(dataset.kgs(), embeddings, output_folder, ptv_name)
    ptv.prepare(dataset.labelled_train_pairs)

    clf = classifiers[classifier_name]()
    eager_ex = Experiment(Eager(clf, ptv, classifier_name))
    results, artifacts = eager_ex.run(dataset)

    prediction_files = write_predictions(
        output_folder, artifacts, dataset, emb_info, classifier_name, ptv_name
    )

    return results, prediction_files


def run_all(
    emb_models: List[str],
    data_paths: List[str],
    folds: List[int],
    sizes: List[int],
    classifier_names: List[str],
    ptv_names: List[str],
):
    for data_path in data_paths:
        for size in sizes:
            for fold in folds:
                for emb_model in emb_models:
                    dataset, emb_info, output_folder = resolve_names(
                        emb_model, data_path, fold, size
                    )
                    results = []
                    for ptv_name in ptv_names:
                        if "Bert" in ptv_name:
                            create_bert_embeddings(dataset, output_folder)
                        for classifier_name in classifier_names:
                            result, _ = run_single(
                                dataset,
                                emb_info,
                                output_folder,
                                classifier_name,
                                ptv_name,
                            )
                            results.append(result)
                    write_result_file(output_folder, dataset, emb_model, results)


def resolve_names(emb_model, data_path, fold, size):
    emb_info = {"boot_ea": boot_ea, "multi_ke": multi_ke, "rdgcn": rdgcn}[emb_model]()
    dataset, emb_info = get_config(data_path, fold, DataSize(size), emb_info)
    existing_folder = find_existing_result_folder(emb_info.model)
    output_folder = existing_folder or emb_info.model.out_folder[:-1]

    return dataset, emb_info, output_folder


def get_embeddings(dataset, emb_info: EmbeddingInfo):
    existing_embedding_folder = find_existing_embedding_folder(emb_info.model, dataset)
    if existing_embedding_folder is not None:
        embeddings = np.load(f"{existing_embedding_folder}/ent_embeds.npy")
    else:
        emb_info.model.set_kgs(dataset.kgs())
        emb_info.model.init()
        emb_info.model.run()
        emb_info.model.test()
        emb_info.model.save()
        embeddings = emb_info.extractor(emb_info.model)

    return embeddings


def find_existing_embedding_folder(embedding_model: BasicModel, dataset: Dataset):
    model_name = type(embedding_model).__name__
    folder = f"data/Embeddings{dataset.data_size.value}K/{model_name}/{dataset.name()}"
    if path.exists(folder):
        dirs = sorted([d for d in glob(f"{folder}/*")])
    else:
        folder = f"../output/results/{model_name}/{dataset.name()}"
        if path.exists(folder):
            dirs = sorted([d for d in glob(f"{folder}/*")])
        else:
            folder = embedding_model.out_folder
            folder = "/".join(folder.split("/")[:-2])
            dirs = sorted(
                [
                    d
                    for d in glob(f"{folder}/*")
                    if embedding_model.args.dataset_division in d
                ]
            )

    if any(dirs):
        return dirs[-1]
    return None


if __name__ == "__main__":
    import torch

    torch.multiprocessing.set_start_method("spawn")
    main(sys.argv)
