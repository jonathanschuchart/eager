import os
import random
import shutil
import subprocess

from openea.modules.args.args_hander import load_args
from openea.modules.load.kgs import read_kgs_from_folder

from dataset.dataset import Dataset


class ScadsDataset(Dataset):
    def __init__(self, data_folder: str, division: str, args_path: str):
        args = load_args(args_path)
        self._data_folder = data_folder
        self.download_and_unzip()
        self._kgs = read_kgs_from_folder(
            data_folder, division, args.alignment_module, args.ordered
        )

        train_links = [(e[0], e[1], 1) for e in self._kgs.train_links]
        valid_links = [(e[0], e[1], 1) for e in self._kgs.valid_links]
        test_links = [(e[0], e[1], 1) for e in self._kgs.test_links]
        super().__init__(
            kg1=self._kgs.kg1,
            kg2=self._kgs.kg2,
            rnd=random.Random(),
            labelled_pairs=train_links + valid_links + test_links,
            # throw them together because of possible inbalance
            # due to removal of inner links
        )
        self._name = data_folder.split("/")[-2] + "/" + division[:-1]

    def download_and_unzip(self):
        if not os.path.exists(self._data_folder):
            parent_dir = self._get_download_dir()
            print(
                f"Did not find movie graph data in {self._data_folder}\nTherefore downloading and creating data"
            )
            repo_path = os.path.join("data", "MovieGraphBenchmark")
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "https://github.com/ScaDS/MovieGraphBenchmark/",
                    repo_path,
                ]
            )
            # execute it as a script
            subprocess.check_call(
                [
                    "python",
                    os.path.join(
                        "data", "MovieGraphBenchmark", "src", "create_graph.py"
                    ),
                ]
            )
            os.makedirs(parent_dir)

            # move files
            ds_dirs = ["imdb-tvdb", "imdb-tmdb", "tmdb-tvdb"]

            for d in ds_dirs:
                shutil.move(
                    os.path.join("data", "MovieGraphBenchmark", "data", d),
                    parent_dir,
                )

            # cleanup
            try:
                shutil.rmtree(repo_path)
            except OSError as e:
                print(f"Error during cleanup for {repo_path}: {e.strerror}")

    def kgs(self):
        return self._kgs

    def name(self):
        return self._name
