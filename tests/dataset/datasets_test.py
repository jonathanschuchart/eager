import os

import pytest
from assertpy import assert_that
from src.dataset.csv_dataset import CsvDataset
from src.dataset.dataset import Dataset, DataSize
from src.dataset.openea_dataset import OpenEaDataset
from src.dataset.scads_dataset import ScadsDataset

OPENEA_DS = [
    "D_W_15K_V1",
    "D_W_15K_V2",
    "D_Y_15K_V1",
    "D_Y_15K_V2",
    "EN_DE_15K_V1",
    "EN_DE_15K_V2",
    "EN_FR_15K_V1",
    "EN_FR_15K_V2",
    "D_W_100K_V1",
    "D_W_100K_V2",
    "D_Y_100K_V1",
    "D_Y_100K_V2",
    "EN_DE_100K_V1",
    "EN_DE_100K_V2",
    "EN_FR_100K_V1",
    "EN_FR_100K_V2",
]

link_size_openea = {
    DataSize.K15.name: {"test": 10500, "train": 3000, "valid": 1500},
    DataSize.K100.name: {"test": 70000, "train": 20000, "valid": 10000},
}

CSV_DS = ["abt-buy", "dblp-acm", "dblp-scholar", "amazon-google"]

MOVIE_DS = ["imdb-tmdb", "imdb-tvdb", "tmdb-tvdb"]


@pytest.mark.parametrize("ds_string", OPENEA_DS)
def test_openea(ds_string):
    data_dir = "data/openea"
    args_json = "bootea_args_15K.json" if "15" in ds_string else "bootea_args_100K.json"
    ds = OpenEaDataset(
        os.path.join(data_dir, ds_string) + os.sep,
        os.path.join("721_5fold", "1") + os.sep,
        os.path.join("..", "OpenEA", "run", "args", args_json),
    )
    size = ds.data_size.name
    assert (
        len(ds.labelled_test_pairs) == link_size_openea[size]["test"]
    ), f"test_pairs assert {len(ds.labelled_test_pairs)} == {link_size_openea[size]['test']} failed for {ds_string}"
    assert (
        len(ds.labelled_val_pairs) == link_size_openea[size]["valid"]
    ), f"val_pairs assert {len(ds.labelled_val_pairs)} == {link_size_openea[size]['valid']} failed for {ds_string}"
    assert (
        len(ds.labelled_train_pairs) == link_size_openea[size]["train"]
    ), f"train_pairs assert {len(ds.labelled_train_pairs)} == {link_size_openea[size]['train']} failed for {ds_string}"
    assert_that(ds.kg1).is_not_none
    assert_that(ds.kg2).is_not_none


def get_expected_labelled(ds: Dataset):
    all_labelled = (
        len(ds.labelled_test_pairs)
        + len(ds.labelled_train_pairs)
        + len(ds.labelled_val_pairs)
    )
    expected_test = int(all_labelled * 0.7)
    expected_train = int(all_labelled * 0.2)
    expected_val = int(all_labelled * 0.1)
    return expected_test, expected_train, expected_val


@pytest.mark.parametrize("ds_string", CSV_DS)
def test_csv_datasets(ds_string):
    data_dir = "data/dbs-er"
    args_json = "bootea_args_15K.json"
    ds = CsvDataset(
        os.path.join(data_dir, ds_string) + os.sep,
        os.path.join("721_5fold", "1") + os.sep,
        os.path.join("..", "OpenEA", "run", "args", args_json),
    )

    expected_test, expected_train, expected_val = get_expected_labelled(ds)
    # allow for some rounding wiggle room
    assert_that(len(ds.labelled_test_pairs)).is_between(
        expected_test - 1, expected_test + 1
    )
    assert_that(len(ds.labelled_train_pairs)).is_between(
        expected_train - 1, expected_train + 1
    )
    assert_that(len(ds.labelled_val_pairs)).is_between(
        expected_val - 1, expected_val + 1
    )
    assert_that(ds.kg1).is_not_none
    assert_that(ds.kg2).is_not_none


@pytest.mark.parametrize("ds_string", MOVIE_DS)
def test_movie_datasets(ds_string):
    data_dir = "data/ScaDS-MB"
    args_json = "bootea_args_15K.json"
    ds = ScadsDataset(
        os.path.join(data_dir, ds_string) + os.sep,
        os.path.join("721_5fold", "1") + os.sep,
        os.path.join("..", "OpenEA", "run", "args", args_json),
    )

    expected_test, expected_train, expected_val = get_expected_labelled(ds)
    # allow for some rounding wiggle room
    assert_that(len(ds.labelled_test_pairs)).is_between(
        expected_test - 1, expected_test + 1
    )
    assert_that(len(ds.labelled_train_pairs)).is_between(
        expected_train - 1, expected_train + 1
    )
    assert_that(len(ds.labelled_val_pairs)).is_between(
        expected_val - 1, expected_val + 1
    )
    assert_that(ds.kg1).is_not_none
    assert_that(ds.kg2).is_not_none
