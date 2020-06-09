from openea.modules.load.kgs import read_kgs_from_folder

from dataset.dataset import Dataset


class OpenEaDataset(Dataset):
    def __init__(self, data_folder: str, division: str):
        self._kgs = read_kgs_from_folder(
            data_folder, division, "mapping", ordered=False
        )
        train_links = [(e[0], e[1], 1) for e in self._kgs.train_links]
        valid_links = [(e[0], e[1], 1) for e in self._kgs.valid_links]
        test_links = [(e[0], e[1], 1) for e in self._kgs.test_links]
        super().__init__(
            self._kgs.kg1, self._kgs.kg2, train_links, valid_links, None, test_links
        )
        self._name = data_folder.split("/")[-2] + "-" + division.replace("/", "-")

    def kgs(self):
        return self._kgs

    def name(self):
        return self._name
