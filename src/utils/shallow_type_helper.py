import glob
import json
import os


def write_typed_dict(dataset_path, out_path):
    type_rels = {}
    for r in ["rel_triples_1", "rel_triples_2"]:
        with open(dataset_path + "/" + r) as rel_file:
            for line in rel_file:
                if "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" in line:
                    trip = line.strip().split("\t")
                    type_rels[trip[0]] = trip[2]
    # for fold in range(1, 6):
    #     type_dict = {}
    #     for test_file in glob.iglob(f"{dataset_path}/721_5fold/{fold}/test_links"):
    #         with open(test_file) as tf:
    #             for line in tf:
    #                 tup = line.strip().split("\t")
    #                 type_dict[tup[0]] = [type_rels[tup[0]]]
    #                 type_dict[tup[1]] = [type_rels[tup[1]]]
    #     foldfolder = f"{out_path}/721_5fold/{fold}/"
    #     if not os.path.exists(foldfolder):
    #         os.makedirs(foldfolder)
    with open(out_path, "w") as fp:
        json.dump(type_rels, fp)


for ds in ["imdb-tvdb", "imdb-tmdb", "tmdb-tvdb"]:
    write_typed_dict(
        f"data/EA-ScaDS-Datasets/ScadsMB/{ds}/",
        f"data/EA-ScaDS-Datasets/ScadsMB/typed_links/datasets/{ds}",
    )
