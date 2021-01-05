import glob
import os
import signal
import sys
import pandas as pd

from tqdm import tqdm

_MODEL_COL = "model_name"
_VECTOR_COL = "vector_name"
_EMBED_COL = "embed_model"
_DS_COL = "dataset"
_LEFT_URI_COL = "left_uri"
_RIGHT_URI_COL = "right_uri"
_FOLD_COL = "fold"
_FN_COL = "fn"
_FP_COL = "fp"
_TN_COL = "tn"
_TP_COL = "tp"
_PREC_COL = "prec"
_REC_COL = "rec"
_FM_COL = "fm"

_DS_NAMES = [
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
    "abt-buy",
    "amazon-google",
    "dblp-acm",
    "dblp-scholar",
    "imdb-tmdb",
    "imdb-tvdb",
    "tmdb-tvdb",
]


def get_pred_files(
    base_folder: str,
    embedding_approach: str,
    dataset_name: str,
    classifier_name: str,
    vector_type: str,
):
    multiple = []
    pred_files = []
    for fold in range(1, 6):
        curr = sorted(
            [
                i
                for i in glob.iglob(
                    f"{base_folder}/output/results_bak/{dataset_name}-721_5fold-{fold}/{embedding_approach}/datasets/*/{dataset_name}-721_5fold-{fold}*_{classifier_name}_{vector_type}_test_pred.csv"
                )
            ]
        )
        if len(curr) > 1:
            multiple.append(curr)
        pred_files.append(curr[-1])
    return pred_files, multiple


def _set_errors(pred, val, exp_pred, exp_val):
    if (pred == exp_pred) & (val == exp_val):
        return 1
    else:
        return 0


def create_pred_df(
    base_folder: str,
    cache_path=None,
    embedding_approaches=["BootEA", "MultiKE", "RDGCN"],
    classifiers=[
        "MLP",
        # "decision tree",
        # "gaussian naive bayes",
        # "random forest 500",
        # "ada boost",
    ],
    vector_type=["OnlyEmb", "OnlySim", "SimAndEmb"],
) -> pd.DataFrame:
    """
    Creates a dataframe from the test pred csv files with calculated errors
    """
    if cache_path is None:
        cache_path = base_folder
    pkl_path = (
        cache_path
        + "_".join(embedding_approaches)
        + "_".join(classifiers)
        + "_".join(vector_type)
        + "_LARGE_"
        + "_df.pkl"
    )
    if os.path.exists(pkl_path):
        print(f"Read cached: {pkl_path}")
        return pd.read_pickle(pkl_path)
    full = []
    multiple = []
    for variables in tqdm(
        [
            (e, c, v, ds)
            for e in embedding_approaches
            for c in classifiers
            for v in vector_type
            for ds in _DS_NAMES
        ],
        desc="Collecting test pred csv files",
    ):
        e, c, v, ds = variables
        fold = 1
        pred_files, m = get_pred_files(base_folder, e, ds, c, v)
        multiple.append(m)
        for pred_file in pred_files:
            multiple.append(m)
            with open(pred_file) as in_file:
                for line in in_file:
                    if "left,right" not in line:
                        row = line.strip().split(",")
                        inner = {
                            _EMBED_COL: e,
                            _MODEL_COL: c,
                            _VECTOR_COL: v,
                            _DS_COL: ds,
                            _LEFT_URI_COL: row[0],
                            _RIGHT_URI_COL: row[1],
                            _FOLD_COL: fold,
                            _FN_COL: _set_errors(int(row[3]), int(row[2]), 0, 1),
                            _FP_COL: _set_errors(int(row[3]), int(row[2]), 1, 0),
                            _TP_COL: _set_errors(int(row[3]), int(row[2]), 1, 1),
                            _TN_COL: _set_errors(int(row[3]), int(row[2]), 0, 0),
                        }
                        full.append(inner)
            fold += 1
    print(
        f"Found {len(multiple)} files with multiple possibilities. Took the newest ones\nCreating df now..."
    )
    df = pd.DataFrame(full)
    pd.to_pickle(df, pkl_path)
    return df


def calculate_measures(df):
    """
    Calculates fm, prec, rec and averages over folds
    """
    summed = (
        df.drop([_LEFT_URI_COL, _RIGHT_URI_COL], axis=1)
        .groupby([_DS_COL, _EMBED_COL, _VECTOR_COL, _MODEL_COL, _FOLD_COL])
        .sum()
    )
    summed[_PREC_COL] = summed[_TP_COL] / (summed[_TP_COL] + summed[_FP_COL])
    summed[_REC_COL] = summed[_TP_COL] / (summed[_TP_COL] + summed[_FN_COL])
    summed[_FM_COL] = 2 * (
        summed[_PREC_COL] * summed[_REC_COL] / (summed[_PREC_COL] + summed[_REC_COL])
    )
    summed = summed[[_PREC_COL, _REC_COL, _FM_COL]]
    return (
        summed.reset_index()
        .groupby([_DS_COL, _EMBED_COL, _VECTOR_COL, _MODEL_COL])
        .mean()
        .drop(_FOLD_COL, axis=1)
        .reset_index()
    )


if __name__ == "__main__":
    import faulthandler

    faulthandler.register(signal.SIGUSR1)
    # base = sys.argv[1]
    base = "/home/jonathan/git/er-embedding-benchmark/"
    dfs = []
    for v in ["OnlyEmb", "OnlySim", "SimAndEmb"]:
        dfs.append(calculate_measures(create_pred_df(base, vector_type=[v],)))
    final = dfs[0].append(dfs[1]).append(dfs[2])
    pd.to_pickle(final, base + "data/largeALL.pkl")
