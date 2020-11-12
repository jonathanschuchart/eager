import pytest
import numpy as np
from assertpy import assert_that
import os

from attribute_features import CartesianCombination, AttributeFeatureCombination
from distance_measures import DateDistance, EmbeddingEuclideanDistance
from matching.pair_to_vec import PairToVec
from similarity_measures import (
    NumberSimilarity,
    Levenshtein,
    TriGram,
    GeneralizedJaccard,
)
from openea.modules.load.kgs import KGs, read_kgs_from_folder


@pytest.fixture
def loaded_kgs():
    path = "data/OpenEA/D_W_15K_V1/"
    if not os.path.exists(path):
        path = os.path.join("..", path)
    return read_kgs_from_folder(
        path, "721_5fold/1/", "mapping", True, remove_unlinked=False,
    )


@pytest.fixture
def alignment(loaded_kgs) -> AttributeFeatureCombination:
    return CartesianCombination(
        loaded_kgs,
        [NumberSimilarity()],
        [DateDistance()],
        [Levenshtein(), TriGram(), GeneralizedJaccard()],
    )


@pytest.fixture
def embedding():
    path = "tests/test_kgs/slice_ent_emb.npy"
    if not os.path.exists(path):
        path = os.path.join("..", path)
    return np.load(path)


def test_align_attrs(alignment):
    e1_attrs = [
        (1, "123"),
        (3, '"123"^^<http://www.w3.org/2001/XMLSchema#double>'),
        (5, "test"),
    ]
    e2_attrs = [
        (1, "123"),
        (4, '"123"^^<http://www.w3.org/2001/XMLSchema#double>'),
        (5, "other"),
    ]
    aligned = alignment._align_attributes(e1_attrs, e2_attrs)
    aligned = [(a.k1, a.k2) for a in aligned]
    assert_that(aligned).contains((1, 1), (5, 5), (3, 4))


def test_align_attrs_complex(alignment):
    e1_attrs = [
        (20, "Canadian ice hockey player"),
        (76, "87th Overall"),
        (36, '"1.9304"^^<http://www.w3.org/2001/XMLSchema#double>'),
        (6, '"1991-12-06"^^<http://www.w3.org/2001/XMLSchema#date>'),
        (46, '"98431.2"^^<http://www.w3.org/2001/XMLSchema#double>'),
        (56, '"2010"^^<http://www.w3.org/2001/XMLSchema#gYear>'),
        (10, '"2012"^^<http://www.w3.org/2001/XMLSchema#gYear>'),
        (8, '"1991"^^<http://www.w3.org/2001/XMLSchema#gYear>'),
        (0, "Melchiori, Julian"),
        (68, "Left"),
    ]
    e2_attrs = [
        (151, "121474"),
        (1, "Canadian ice hockey player"),
        (11, '"1991-12-06"^^<http://www.w3.org/2001/XMLSchema#date>'),
        (157, "73778"),
    ]
    aligned = alignment._align_attributes(e1_attrs, e2_attrs)
    aligned = [(a.k1, a.k2) for a in aligned]
    assert_that(aligned).contains(
        (20, 151),
        (20, 1),
        (20, 157),
        (76, 151),
        (76, 1),
        (76, 157),
        (0, 151),
        (0, 1),
        (0, 157),
        (68, 151),
        (68, 1),
        (68, 157),
        (6, 11),
    )


def test_calculate_from_embeddings_with_training(loaded_kgs, embedding, alignment):
    pvp = PairToVec(
        embedding, loaded_kgs, "some_name", alignment, [EmbeddingEuclideanDistance()]
    )
    similarities = pvp._calculate_pair_comparisons(0, 12)
    assert_that(similarities).contains_key(
        "Levenshtein.30:30",
        "GeneralizedJaccard.30:30",
        "TriGram.30:30",
        "NumberSimilarity.38:38",
        "NumberSimilarity.40:40",
        "NumberSimilarity.60:60",
        # "NumberDist.48:60",
        # "NumberDist.40:60",
        # "Lev.28:136",
        # "GenJac.28:136",
        # "Trigram.28:136",
        # "Lev.28:164",
        # "GenJac.28:164",
        # "Trigram.28:164",
        "Levenshtein.0:0",
        "GeneralizedJaccard.0:0",
        "TriGram.0:0",
        "EmbeddingEuclideanDistance",
    )
    assert_that(similarities["Levenshtein.0:0"]).is_greater_than(0.0)
