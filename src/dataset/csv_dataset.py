import random
from enum import Enum

from openea.modules.args.args_hander import load_args
from openea.modules.load.kgs import read_kgs_from_folder

from dataset.dataset import Dataset


class CsvType(Enum):
    products = 1
    articles = 2


class CsvDataset(Dataset):
    _download_urls = [
        "https://speicherwolke.uni-leipzig.de/index.php/s/ro25Tb4WdB6Y4iJ/download"
    ]
    _zip_names = ["DBS-ER.zip"]

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
            labelled_pairs=train_links,
            labelled_val_pairs=valid_links,
            labelled_test_pairs=test_links,
        )
        self._name = data_folder.split("/")[-2] + "/" + division[:-1]

    def kgs(self):
        return self._kgs

    def name(self):
        return self._name
