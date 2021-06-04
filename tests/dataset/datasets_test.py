import os

import pytest
from assertpy import assert_that
from src.dataset.dataset import DataSize
from src.dataset.openea_dataset import OpenEaDataset

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
    print(args_json)
    print(size)
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
