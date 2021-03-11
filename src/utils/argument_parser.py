import argparse

openea_data_names = ["D_W", "D_Y", "EN_DE", "EN_FR"]
movie_data_names = ["imdb-tmdb", "imdb-tvdb", "tmdb-tvdb"]
csv_data_names = ["abt-buy", "amazon-google", "dblp-acm", "dblp-scholar"]


def parse_arguments(argv):
    data_names = openea_data_names + movie_data_names + csv_data_names
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emb_models",
        nargs="+",
        default=["boot_ea", "multi_ke", "rdgcn"],
        choices=["boot_ea", "multi_ke", "rdgcn"],
        required=False,
        help="The KG embedding models to use.",
    )
    parser.add_argument(
        "--data_names", nargs="+", default=data_names, choices=data_names
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(range(1, 6)),
        choices=list(range(1, 6)),
        required=False,
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[15, 100],
        choices=[15, 100],
        required=False,
        help="The sizes of the datasets to use. Only applies to OpenEA datasets.",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        type=int,
        default=[1, 2],
        choices=[1, 2],
        help="The versions of datasets to use. Only applies to OpenEA datasets.",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["RF", "MLP"],
        choices=["RF", "MLP"],
        required=False,
    )
    parser.add_argument(
        "--ptv_names",
        nargs="+",
        default=["SimConcatAndEmb", "OnlySimConcat", "OnlyEmb"],
        choices=[
            "OnlyEmb",
            "SimAndEmb",
            "OnlySim",
            "SimConcatAndEmb",
            "OnlySimConcat",
            "BertConcatAndEmb",
            "OnlyBertConcat",
        ],
        required=False,
        help="The type of EAGER to use. I.e. how the input to the classifier is created.",
    )
    args = parser.parse_args()

    args.data_paths = resolve_data_paths(args.data_names, args.sizes, args.versions)

    return args


def resolve_data_paths(data_names, sizes, versions):
    paths = [
        resolve_data_path(name, size, version)
        for name in data_names
        for size in sizes
        for version in versions
    ]
    # remove potential duplicates
    return list(dict.fromkeys(paths))


def resolve_data_path(data_name, size, version):
    if data_name in csv_data_names:
        return f"../datasets/{data_name}/"
    if data_name in movie_data_names:
        return f"../datasets/ScadsMB/{data_name}/"
    if data_name in openea_data_names:
        return f"../datasets/{data_name}_{size}K_V{version}/"
