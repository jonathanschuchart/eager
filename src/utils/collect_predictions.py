import zipfile
import os
import glob


def is_numeric_path(s: str):
    s = s.split("/")[-1]
    try:
        int(s[-14:])
        return True
    except:
        return False


classifier_reject_list = [
    "_logistic regression_",
    "_svc_",
    "_random forest 20_",
    "_random forest 50_",
    "_random forest 100_",
    "_random forest 200_",
    "_decision tree_",
]


def expand_paths(cur_path):
    if os.path.isdir(cur_path):
        next_paths = glob.glob(f"{cur_path}/*")
        if all(is_numeric_path(p) for p in next_paths) and len(next_paths) > 1:
            raise Exception(f"Found more than one result folder in {cur_path}")
        return [
            child for path in glob.glob(f"{cur_path}/*") for child in expand_paths(path)
        ]
    return (
        [cur_path]
        if ".csv" == cur_path[-4:]
        and "_100K_" in cur_path
        and not any(c in cur_path for c in classifier_reject_list)
        and "Normalized_" not in cur_path
        else []
    )


def main():
    """
    Collects all prediction files from all experiments into a zip file
    :return:
    """
    all_pred_files = expand_paths("../../output/results")
    with zipfile.ZipFile(
        "predictions_100K.zip", "w", compression=zipfile.ZIP_LZMA
    ) as zip:
        for pred_file in all_pred_files:
            zip.write(pred_file)


if __name__ == "__main__":
    main()
