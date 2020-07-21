import glob
import os
import pandas as pd


def is_numeric_path(s: str):
    s = s.split("/")[-1]
    try:
        int(s[-14:])
        return True
    except:
        return False


def expand_paths(cur_path):
    if os.path.isdir(cur_path):
        next_paths = glob.glob(f"{cur_path}/*")
        if all(is_numeric_path(p) for p in next_paths) and len(next_paths) > 1:
            raise Exception(f"Found more than one result folder in {cur_path}")
        return [
            child for path in glob.glob(f"{cur_path}/*") for child in expand_paths(path)
        ]
    return [cur_path] if ".csv" == cur_path[-4:] else []


def main():
    paths = expand_paths("output/results")
    split_paths = [path.split("/") for path in paths]
    index = [(path[2], path[3]) for path in split_paths]
    dfs = [
        pd.read_csv(path).assign(dataset=dataset, embed_model=model)
        for path, (dataset, model) in zip(paths, index)
    ]
    pd.concat(dfs).to_csv("all_results.csv", index=False)

    print("\n".join(sorted(paths)))
    print(len(paths))


if __name__ == "__main__":
    main()
