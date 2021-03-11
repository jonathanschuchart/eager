import random

import pandas as pd
import numpy as np
import os
from openea.approaches import BootEA
from openea.modules.args.args_hander import load_args

from dataset.csv_dataset import CsvDataset, CsvType
from dataset.scads_dataset import ScadsDataset
import csv


base_path = "../datasets"


def to_csv(tuples, path):
    base_path = os.path.dirname(path)
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    pd.DataFrame(tuples).to_csv(
        path, header=False, index=False, sep="\t", quoting=csv.QUOTE_NONE
    )


def split_dataset(dataset, path):
    to_csv(dataset.kg1.attribute_triples_list, f"{path}/attr_triples_1")
    to_csv(dataset.kg2.attribute_triples_list, f"{path}/attr_triples_2")

    to_csv(dataset.kg1.relation_triples_list, f"{path}/rel_triples_1")
    to_csv(dataset.kg2.relation_triples_list, f"{path}/rel_triples_2")

    all_links = (
        dataset.kgs().uri_train_links
        + dataset.kgs().uri_valid_links
        + dataset.kgs().uri_test_links
    )
    all_links = [t if len(t) == 2 else t[:2] for t in all_links]
    to_csv(all_links, f"{path}/ent_links")

    part_len = len(all_links) / 10
    np.random.seed(42)
    np.random.shuffle(all_links)

    if not os.path.exists(f"{path}/721_5fold/"):
        os.mkdir(f"{path}/721_5fold/")

    def combine_slices(start, end):
        if start < end:
            return all_links[start:end]
        else:
            return all_links[:end] + all_links[start:]

    for fold in range(5):
        test_start = int(2 * fold * part_len)
        test_end = int(((2 * fold + 7) % 10) * part_len)
        test_links = combine_slices(test_start, test_end)

        train_start = int(((2 * fold + 7) % 10) * part_len)
        train_end = int(((2 * fold + 9) % 10) * part_len)
        train_links = combine_slices(train_start, train_end)

        valid_start = int(((2 * fold + 9) % 10) * part_len)
        valid_end = int(2 * fold * part_len)
        valid_links = combine_slices(valid_start, valid_end)

        to_csv(test_links, f"{path}/721_5fold/{fold + 1}/test_links")
        to_csv(train_links, f"{path}/721_5fold/{fold + 1}/train_links")
        to_csv(valid_links, f"{path}/721_5fold/{fold + 1}/valid_links")


def split_scads(source1, source2, rnd: random.Random):
    model = BootEA()
    model.set_args(load_args("../OpenEA/run/args/bootea_args_15K.json"))
    dataset = ScadsDataset(f"{base_path}/ScadsMB/100/", source1, source2, model, rnd)
    split_dataset(dataset, f"{base_path}/ScadsMB/{source1}-{source2}")


def split_abt_buy(rnd: random.Random):
    model = BootEA()
    model.set_args(load_args("../OpenEA/run/args/bootea_args_15K.json"))
    dataset = CsvDataset(
        CsvType.products,
        f"{base_path}/abt-buy/Abt.csv",
        f"{base_path}/abt-buy/Buy.csv",
        f"{base_path}/abt-buy/abt_buy_perfectMapping.csv",
        model,
        rnd,
    )
    print(f"abt-buy: {len(dataset.kg1.entities_set)}, {len(dataset.kg2.entities_set)}")
    split_dataset(dataset, f"{base_path}/abt-buy")


def split_amazon_google(rnd: random.Random):
    model = BootEA()
    model.set_args(load_args("../OpenEA/run/args/bootea_args_15K.json"))
    dataset = CsvDataset(
        CsvType.products,
        f"{base_path}/amazon-google/Amazon.csv",
        f"{base_path}/amazon-google/GoogleProducts.csv",
        f"{base_path}/amazon-google/Amzon_GoogleProducts_perfectMapping.csv",
        model,
        rnd,
    )
    print(
        f"amazon-google: {len(dataset.kg1.entities_set)}, {len(dataset.kg2.entities_set)}"
    )
    split_dataset(dataset, f"{base_path}/amazon-google")


def split_dblp_acm(rnd: random.Random):
    model = BootEA()
    model.set_args(load_args("../OpenEA/run/args/bootea_args_15K.json"))
    dataset = CsvDataset(
        CsvType.articles,
        f"{base_path}/dblp-acm/DBLP2.csv",
        f"{base_path}/dblp-acm/ACM.csv",
        f"{base_path}/dblp-acm/DBLP-ACM_perfectMapping.csv",
        model,
        rnd,
    )
    print(
        f"dblp2-acm: {len(dataset.kg1.entities_set)}, {len(dataset.kg2.entities_set)}"
    )
    split_dataset(dataset, f"{base_path}/dblp-acm")


def split_dblp_scholar(rnd: random.Random):
    model = BootEA()
    model.set_args(load_args("../OpenEA/run/args/bootea_args_15K.json"))
    dataset = CsvDataset(
        CsvType.articles,
        f"{base_path}/dblp-scholar/DBLP1.csv",
        f"{base_path}/dblp-scholar/Scholar.csv",
        f"{base_path}/dblp-scholar/DBLP-Scholar_perfectMapping.csv",
        model,
        rnd,
    )
    print(
        f"dblp-scholar: {len(dataset.kg1.entities_set)}, {len(dataset.kg2.entities_set)}"
    )
    split_dataset(dataset, f"{base_path}/dblp-scholar")


def check_split(base_path):
    def read_file_to_set(path):
        with open(path) as f:
            return set(tuple(line.split("\t")) for line in f.readlines())

    train_links_per_fold = [
        read_file_to_set(f"{base_path}/721_5fold/{i}/train_links") for i in range(1, 6)
    ]
    valid_links_per_fold = [
        read_file_to_set(f"{base_path}/721_5fold/{i}/valid_links") for i in range(1, 6)
    ]
    test_links_per_fold = [
        read_file_to_set(f"{base_path}/721_5fold/{i}/test_links") for i in range(1, 6)
    ]
    all_links = read_file_to_set(f"{base_path}/ent_links")

    for fold in range(5):
        train = train_links_per_fold[fold]
        valid = valid_links_per_fold[fold]
        test = test_links_per_fold[fold]
        assert not any(train.intersection(valid))
        assert not any(train.intersection(test))
        assert not any(test.intersection(valid))
        assert len(all_links) == len(train) + len(valid) + len(test)

    for fold_list, name, allowed_overlap in [
        (train_links_per_fold, "train", 0),
        (valid_links_per_fold, "valid", 0),
        (test_links_per_fold, "test", np.math.ceil(0.5 * len(all_links))),
    ]:
        for i, set1 in enumerate(fold_list[:-1]):
            for k, set2 in enumerate(fold_list[i + 1 :]):
                assert (
                    len(set1.intersection(set2)) <= allowed_overlap
                ), f"{name} part failed allowed overlap of {allowed_overlap} between folds {i} and {k}"


def main():
    rnd = lambda: random.Random(42)
    split_abt_buy(rnd())
    check_split(f"{base_path}/abt-buy")

    split_amazon_google(rnd())
    check_split(f"{base_path}/amazon-google")

    split_dblp_acm(rnd())
    check_split(f"{base_path}/dblp-acm")

    split_dblp_scholar(rnd())
    check_split(f"{base_path}/dblp-scholar")

    split_scads("imdb", "tmdb", rnd())
    check_split(f"{base_path}/ScadsMB/imdb-tmdb")

    split_scads("imdb", "tvdb", rnd())
    check_split(f"{base_path}/ScadsMB/imdb-tvdb")

    split_scads("tmdb", "tvdb", rnd())
    check_split(f"{base_path}/ScadsMB/tmdb-tvdb")


if __name__ == "__main__":
    main()
