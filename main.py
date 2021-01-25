import random
from glob import glob
from os import path

import numpy as np
from openea.models.basic_model import BasicModel
from sentence_transformers import SentenceTransformer

from attribute_features import CartesianCombination, AllToOneCombination, _remove_type
from dataset.dataset import Dataset
from distance_measures import DateDistance, EmbeddingEuclideanDistance
from eager import Eager
from experiment import Experiments, Experiment
from matching.pair_to_vec import PairToVec
from prepared_models import classifier_factories
from run_configs import configs, EmbeddingInfo
from similarity_measures import (
    Levenshtein,
    GeneralizedJaccard,
    TriGram,
    EmbeddingConcatenation,
    NumberSimilarity,
    BertFeatureSimilarity,
    BertCosineSimilarity,
    BertConcatenation,
    bert_embed,
)
from write_results import find_existing_result_folder, write_result_files


def main():
    for dataset, emb_info in configs():
        existing_folder = find_existing_result_folder(emb_info.model)
        output_folder = existing_folder or emb_info.model.out_folder[:-1]

        run_for_dataset(dataset, emb_info, output_folder)
        # create_bert_embeddings(dataset, output_folder)
        # fix_bert_embeddings(dataset, output_folder)


def fix_bert_embeddings(dataset, output_folder):
    kgs = dataset.kgs()
    new_embeds_1 = []
    embeds_1 = np.load(path.join(output_folder, "bert_embeds_1.npy"), allow_pickle=True)
    for e1, _, embed in embeds_1:
        v1 = " ".join(_remove_type(v) for _, v in sorted(kgs.kg1.av_dict[e1]))
        new_embeds_1.append((e1, v1, embed))
    new_embeds_2 = []
    embeds_2 = np.load(path.join(output_folder, "bert_embeds_2.npy"), allow_pickle=True)
    for e2, _, embed in embeds_2:
        v2 = " ".join(_remove_type(v) for _, v in sorted(kgs.kg2.av_dict[e2]))
        new_embeds_2.append((e2, v2, embed))
    np.save(path.join(output_folder, "bert_embeds_1"), new_embeds_1)
    np.save(path.join(output_folder, "bert_embeds_2"), new_embeds_2)


def create_bert_embeddings(dataset, output_folder):
    kgs = dataset.kgs()
    bert_key = "distilbert-multilingual-nli-stsb-quora-ranking"
    bert_model = SentenceTransformer(bert_key)
    kg1_dict = kgs.kg1.av_dict
    bert_embeds_1 = []
    for e1, e1_attrs in kg1_dict.items():
        v1 = " ".join(_remove_type(v) for _, v in sorted(e1_attrs))
        bert_embeds_1.append((e1, v1, bert_embed(bert_model, v1)))

    kg2_dict = kgs.kg2.av_dict
    bert_embeds_2 = []
    for e2, e2_attrs in kg2_dict.items():
        v2 = " ".join(_remove_type(v) for _, v in sorted(e2_attrs))
        bert_embeds_2.append((e2, v2, bert_embed(bert_model, v2)))

    np.save(path.join(output_folder, "bert_embeds_1"), bert_embeds_1)
    np.save(path.join(output_folder, "bert_embeds_2"), bert_embeds_2)


def run_for_dataset(dataset, emb_info, output_folder):
    import tensorflow as tf

    tf.reset_default_graph()

    rnd = random.Random(42)
    dataset.add_negative_samples(rnd)
    print(f"using {emb_info.name} on {dataset.name()}")
    embeddings = get_embeddings(dataset, emb_info)

    kgs = dataset.kgs()
    cartesian_attr_combination = CartesianCombination(
        kgs,
        [NumberSimilarity()],
        [DateDistance()],
        [Levenshtein(), GeneralizedJaccard(), TriGram()],
    )
    no_attribute_combinations = CartesianCombination(kgs, [], [], [])

    all_to_one_concat = AllToOneCombination(
        kgs, [Levenshtein(), GeneralizedJaccard(), TriGram()]
    )

    bert_concat = AllToOneCombination(
        kgs, [BertConcatenation(output_folder), BertCosineSimilarity(output_folder)]
    )

    embedding_measures = [EmbeddingEuclideanDistance(), EmbeddingConcatenation()]
    support_threshold = 0.1
    pair_to_vecs = [
        # lambda: PairToVec(
        #     embeddings,
        #     kgs,
        #     "SimAndEmb",
        #     cartesian_attr_combination,
        #     embedding_measures,
        #     support_threshold,
        # ),
        # lambda: PairToVec(
        #     embeddings,
        #     kgs,
        #     "OnlyEmb",
        #     no_attribute_combinations,
        #     [EmbeddingConcatenation()],
        #     support_threshold,
        # ),
        # lambda: PairToVec(
        #     embeddings,
        #     kgs,
        #     "OnlySim",
        #     cartesian_attr_combination,
        #     [],
        #     support_threshold,
        # ),
        lambda: PairToVec(
            embeddings, kgs, "BertConcatAndEmb", bert_concat, embedding_measures
        ),
        lambda: PairToVec(embeddings, kgs, "OnlyBertConcat", bert_concat, []),
        # lambda: PairToVec(
        #     embeddings, kgs, "AllDiffAndEmb", all_to_one_diff, embedding_measures
        # ),
        # lambda: PairToVec(embeddings, kgs, "OnlyAllDiff", all_to_one_diff, []),
    ]
    results_list = []
    for pvp in pair_to_vecs:
        pvp = pvp()
        pvp.prepare(dataset.labelled_train_pairs)
        experiments = Experiments(
            output_folder,
            [
                Experiment(Eager(classifier_fac(), pvp, name))
                # for pair_to_vec in pair_to_vecs
                for name, classifier_fac in classifier_factories
            ],
            dataset,
            None,
        )

        results_list.extend(experiments.run())
    write_result_files(output_folder, dataset, emb_info.name, results_list)


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
    folder = f"data/Embeddings{dataset.data_size.value}K/{type(embedding_model).__name__}/{dataset.name()}"
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


if __name__ == "__main__":
    import torch

    torch.multiprocessing.set_start_method("spawn")
    main()
