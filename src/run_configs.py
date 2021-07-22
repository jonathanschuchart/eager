from openea.approaches import RDGCN, BootEA, MultiKE
from openea.modules.args.args_hander import load_args

from dataset.csv_dataset import CsvDataset
from dataset.dataset import DataSize
from dataset.openea_dataset import OpenEaDataset
from dataset.scads_dataset import ScadsDataset


class EmbeddingInfo:
    def __init__(self, model, name, extractor):
        self.model = model
        self.name = name
        self.extractor = extractor


def boot_ea() -> EmbeddingInfo:
    model = BootEA()
    model.set_args(load_args("../OpenEA/run/args/bootea_args_15K.json"))
    return EmbeddingInfo(
        model, "bootea", lambda m: m.ent_embeds.eval(session=m.session)
    )


def rdgcn() -> EmbeddingInfo:
    model = RDGCN()
    model.set_args(load_args("../OpenEA/run/args/rdgcn_args_15K.json"))
    return EmbeddingInfo(model, "rdgcn", lambda m: m.sess.run(m.output))


def multi_ke() -> EmbeddingInfo:
    model = MultiKE()
    args = load_args("../OpenEA/run/args/multike_args_15K.json")
    args.word2vec_path = args.word2vec_path[3:]
    model.set_args(args)
    return EmbeddingInfo(
        model, "multike", lambda m: m.ent_embeds.eval(session=m.session)
    )


def get_config(path: str, division: int, size: DataSize, emb_info: EmbeddingInfo):
    ds_cls = OpenEaDataset
    if "imdb" in path or "tmdb" in path:
        ds_cls = ScadsDataset
    if "abt-buy" in path or "amazon" in path or "dblp" in path:
        ds_cls = CsvDataset

    dataset = ds_cls(
        path,
        f"721_5fold/{division}/",
        f"../OpenEA/run/args/{type(emb_info.model).__name__.lower()}_args_{size.value}K.json",
    )
    emb_info.model.args.output = f"output/results/{dataset.name().replace('/', '-')}/"
    emb_info.model.set_args(emb_info.model.args)
    return dataset, emb_info
