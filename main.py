import random
from typing import List, Tuple, Dict
import numpy as np
import os

from rdf2vec.converters import rdflib_to_kg
from rdf2vec.graph import KnowledgeGraph

from src.dataset.ScadsDataset import ScadsDataset
from src.dataset.dataset import Dataset
from src.embedding.rdf2vec import Rdf2Vec, Rdf2VecConfig
from src.knowledge_graph.csv import convert_products_to_rdf
from src.matching.mlp import MLP
from src.matching.torch import TorchModelTrainer
from src.quality.quality_measures import (
    get_confusion_matrix,
    precision,
    recall,
    fmeasure,
)


def mutag_example():
    data = Dataset()
    data.load_train_test(
        "data/mutag/mutag.owl",
        "data/mutag/MUTAG_train.tsv",
        "data/mutag/MUTAG_test.tsv",
    )

    rdf = Rdf2Vec(data.kg, {"sg": 2})
    rdf.fit(data.all_entities)
    embeddings = rdf.embed(data.all_entities)
    print(embeddings[:10])


def get_entities(entity_type, dataset_name):
    entities = []
    with open(f"data/ScadsMB/100/Curated/{entity_type}") as f:
        for line in f.readlines():
            entries = line.split(",")
            for e in entries:
                if e[45:49] == dataset_name:
                    entities.append(e)
    return entities


def get_positive_pairs(entity_type, dataset_names) -> List[Tuple[str, str]]:
    pairs = []
    with open(f"data/ScadsMB/100/Curated/{entity_type}") as f:
        for line in f.readlines():
            entries = line.split(",")
            pair = tuple([e for e in entries if e[45:49] in dataset_names])
            if len(pair) == len(dataset_names):
                pairs.append(pair)
    return pairs


def sample_negative_from_list(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    e1s = list({e1 for e1, _ in pairs})
    e2s = list({e2 for _, e2 in pairs})
    negative_pairs = []
    rnd = random.Random()
    while len(negative_pairs) < len(pairs):
        e1 = rnd.choice(e1s)
        e2 = rnd.choice(e2s)
        if (e1, e2) not in pairs and (e1, e2) not in negative_pairs:
            negative_pairs.append((e1, e2))
    return negative_pairs


def sample_negative(pairs: Dict[str, List[Tuple[str, str]]]) -> List[Tuple[str, str]]:
    type_internal_pairs = [
        p for neg_pairs in pairs.values() for p in sample_negative_from_list(neg_pairs)
    ]
    any_pairs = [
        p
        for p in sample_negative_from_list(
            [p for neg_pairs in pairs.values() for p in neg_pairs]
        )
    ]
    return type_internal_pairs + any_pairs


def load_scads_kg(
    dataset_name: str, e_type_list: List[str], emb_dim=200
) -> Tuple[KnowledgeGraph, List[str], List[np.array]]:
    kg = rdflib_to_kg(
        f"data/ScadsMB/100/{dataset_name.lower()}_snippet.nt", filetype="nt"
    )
    entities = [
        e for e_type in e_type_list for e in get_entities(e_type, dataset_name.upper())
    ]
    if os.path.exists(f"embeddings_{dataset_name}.npy"):
        embeddings = np.load(f"embeddings_{dataset_name}.npy")
    else:
        rdf = Rdf2Vec(kg, Rdf2VecConfig(embedding_size=emb_dim, sg=1, max_iter=1000))
        rdf.fit(entities)
        embeddings = rdf.embed(entities)
    np.save(f"embeddings_{dataset_name}.npy", embeddings)
    print(f"Created embeddings for {dataset_name}")
    return kg, entities, embeddings


def episode_example(sources: List[str], e_type_list: List[str]):
    imdb_kg, imdb_entities, imdb_embeddings = load_scads_kg("imdb", e_type_list)
    tmdb_kg, tmdb_entities, tmdb_embeddings = load_scads_kg("tmdb", e_type_list)
    tvdb_kg, tvdb_entities, tvdb_embeddings = load_scads_kg("tvdb", e_type_list)
    dataset = ScadsDataset("data/ScadsMB/100", sources, e_type_list)
    embedding_lookup = {e: emb for e, emb in zip(imdb_entities, imdb_embeddings)}
    embedding_lookup.update({e: emb for e, emb in zip(tmdb_entities, tmdb_embeddings)})
    embedding_lookup.update({e: emb for e, emb in zip(tvdb_entities, tvdb_embeddings)})
    trainer = TorchModelTrainer(MLP([400, 50, 2]), 20, 1000)
    trainer.fit(dataset.train_pairs, dataset.val_pairs, embedding_lookup)
    result = trainer.predict(dataset.test_pairs, embedding_lookup)
    pred_pos = {
        (e[0], e[1])
        for e, res in zip(dataset.test_pairs, result)
        if e[2] == np.argmax(res)
    }
    conf_mat = get_confusion_matrix(
        {(e[0], e[1]) for e in dataset.test_pairs if e[2] == 1},
        {(e[0], e[1]) for e in dataset.test_pairs if e[2] == 0},
        pred_pos,
    )
    print(precision(conf_mat), recall(conf_mat), fmeasure(conf_mat))


def main():
    # episode_example(
    #     ["imdb", "tvdb", "tmdb"], ["episode", "person", "movie", "tvSeries", "company"]
    # )
    # print(
    #     convert_products_to_rdf("data/amazon-google/GoogleProducts.csv")
    #     .serialize(format="turtle")
    #     .decode("utf-8")
    # )
    # print(
    #     convert_products_to_rdf("data/amazon-google/Amazon.csv", "title")
    #     .serialize(format="turtle")
    #     .decode("utf-8")
    # )
    print(
        convert_products_to_rdf("data/abt-buy/Abt.csv")
        .serialize(format="turtle")
        .decode("utf-8")
    )
    print(
        convert_products_to_rdf("data/abt-buy/Buy.csv")
        .serialize(format="turtle")
        .decode("utf-8")
    )


if __name__ == "__main__":
    main()
