import random
from typing import List, Tuple

from gensim.models import Word2Vec
from rdf2vec.converters import rdflib_to_kg

from src.dataset.dataset import Dataset
from src.embedding.rdf2vec import Rdf2Vec
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


def sample_negative(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    e1s = list({e1 for e1, _ in pairs})
    e2s = list({e2 for _, e2 in pairs})
    negative_pairs = []
    rnd = random.Random()
    while len(negative_pairs) < len(pairs):
        e1 = rnd.choice(e1s)
        e2 = rnd.choice(e2s)
        if (e1, e2) not in pairs and (e1, e2) not in negative_pairs:
            negative_pairs.append((e1, e2))

    return list(negative_pairs)


def episode_example():
    imdb_kg = rdflib_to_kg("data/ScadsMB/100/imdb_snippet.nt", filetype="nt")
    imdb_entities = get_entities("episode", "IMDB")
    imdb_rdf = Rdf2Vec(imdb_kg, {"embedding_size": 200, "sg": 1})
    imdb_rdf.fit(imdb_entities)
    # model = Word2Vec.load("../../Downloads/wikid2Vec_cbow_200_5_5_4_500")
    # imdb_rdf._transformer.model_ = model
    imdb_embeddings = imdb_rdf.embed(imdb_entities)

    tmdb_kg = rdflib_to_kg("data/ScadsMB/100/tmdb_snippet.nt", filetype="nt")
    tmdb_entities = get_entities("episode", "TMDB")
    tmdb_rdf = Rdf2Vec(tmdb_kg, {"embedding_size": 200, "sg": 1})
    tmdb_rdf.fit(tmdb_entities)
    tmdb_embeddings = tmdb_rdf.embed(tmdb_entities)

    pairs = get_positive_pairs("episode", ["IMDB", "TMDB"])
    negative_pairs = sample_negative(pairs)

    labelled_pairs = [(e1, e2, 1) for e1, e2 in pairs] + [
        (e1, e2, 0) for e1, e2 in negative_pairs
    ]
    embedding_lookup = {e: emb for e, emb in zip(imdb_entities, imdb_embeddings)}
    embedding_lookup.update({e: emb for e, emb in zip(tmdb_entities, tmdb_embeddings)})
    trainer = TorchModelTrainer(MLP([400, 50, 2]), 1000, 1000)
    trainer.fit(labelled_pairs, embedding_lookup)
    all_pairs = pairs + negative_pairs
    result = trainer.predict(all_pairs, embedding_lookup)
    pred_pos = {(e1, e2) for (e1, e2), res in zip(all_pairs, result) if res[0] < res[1]}
    conf_mat = get_confusion_matrix(set(pairs), set(negative_pairs), pred_pos)
    print(precision(conf_mat), recall(conf_mat), fmeasure(conf_mat))


def main():
    episode_example()


if __name__ == "__main__":
    main()
