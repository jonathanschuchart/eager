import random

from openea.modules.args.args_hander import load_args
from openea.modules.load.kgs import read_kgs_from_folder

from dataset.dataset import Dataset, DataSize


class OpenEaDataset(Dataset):
    _download_urls = [
        "https://www.dropbox.com/s/nzjxbam47f9yk3d/OpenEA_dataset_v1.1.zip?dl=1"
    ]
    _zip_names = ["OpenEA_dataset_v1.1.zip"]

    def __init__(self, data_folder: str, division: str, args_path: str):
        args = load_args(args_path)
        self.data_size = DataSize.K100 if "100" in args_path else DataSize.K15
        self._data_folder = data_folder
        self.download_and_unzip()
        self._kgs = read_kgs_from_folder(
            data_folder, division, args.alignment_module, args.ordered
        )
        train_links = [(e[0], e[1], 1) for e in self._kgs.train_links]
        valid_links = [(e[0], e[1], 1) for e in self._kgs.valid_links]
        test_links = [(e[0], e[1], 1) for e in self._kgs.test_links]
        super().__init__(
            self._kgs.kg1,
            self._kgs.kg2,
            random.Random(),
            train_links,
            valid_links,
            None,
            test_links,
        )
        self._name = data_folder.split("/")[-2] + "/" + division[:-1]

    def kgs(self):
        return self._kgs

    def name(self):
        return self._name
