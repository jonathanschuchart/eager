import pickle
import pytest
import numpy as np
from assertpy import assert_that
from similarity.similarities import (
    calculate_from_embeddings,
    calculate_from_embeddings_with_training,
    align_attributes,
)


@pytest.fixture
def loaded_kgs():
    return pickle.load(open("src/similarity/tests/test_kgs/kgs.pkl", "rb"))


@pytest.fixture
def embedding():
    return np.load("src/similarity/tests/test_kgs/slice_ent_emb.npy")


def test_align_attrs():
    e1_attrs = {
        1: "123",
        3: '"123"^^<http://www.w3.org/2001/XMLSchema#double>',
        5: "test",
    }
    e2_attrs = {
        1: "123",
        4: '"123"^^<http://www.w3.org/2001/XMLSchema#double>',
        5: "other",
    }

    assert [(1, 1), (5, 5)] == align_attributes(e1_attrs, e2_attrs)


def test_calculate_from_embeddings(loaded_kgs, embedding):
    similarities = calculate_from_embeddings(embedding, loaded_kgs, 5, "euclidean")
    assert_that(similarities).does_not_contain_key((0, 0))
    assert_that(similarities[(0, 12)]).contains_key(
        "Lev.0:0",
        "GenJac.0:0",
        "Trigram.0:0",
        "DateDist.73:73",
        "NumberDist.125:125",
        "euclidean",
    )
    assert_that(similarities[(0, 12)]["Lev.0:0"]).is_greater_than(0.0)


def test_calculate_from_embeddings_with_training(loaded_kgs, embedding):
    similarities = calculate_from_embeddings_with_training(
        embedding, [(0, 12, 1)], loaded_kgs, "euclidean"
    )
    assert_that(similarities).does_not_contain_key((0, 0))
    assert_that(similarities[(0, 12)]).contains_key(
        "Lev.0:0",
        "GenJac.0:0",
        "Trigram.0:0",
        "DateDist.73:73",
        "NumberDist.125:125",
        "euclidean",
    )
    assert_that(similarities[(0, 12)]["Lev.0:0"]).is_greater_than(0.0)
