from openea.approaches import BootEA, RDGCN, MultiKE
from openea.modules.args.args_hander import load_args

from dataset.dataset import DataSize
from dataset.openea_dataset import OpenEaDataset


class EmbeddingInfo:
    def __init__(self, model, name, extractor):
        self.model = model
        self.name = name
        self.extractor = extractor


def boot_ea() -> EmbeddingInfo:
    model = BootEA()
    model.set_args(load_args("../OpenEA/run/args/bootea_args_15K.json"))
    return EmbeddingInfo(
        model, "boot_ea", lambda m: m.ent_embeds.eval(session=m.session)
    )


def rdgcn() -> EmbeddingInfo:
    model = RDGCN()
    model.set_args(load_args("../OpenEA/run/args/rdgcn_args_15K.json"))
    return EmbeddingInfo(model, "rdgcn", lambda m: m.sess.run(m.output))


def multi_ke() -> EmbeddingInfo:
    model = MultiKE()
    model.set_args(load_args("../OpenEA/run/args/multike_args_15K.json"))
    return EmbeddingInfo(
        model, "multi_ke", lambda m: m.ent_embeds.eval(session=m.session)
    )


# Datasets:
def abt_buy(division: int, model: EmbeddingInfo):
    return get_config(f"data/abt-buy/", division, DataSize.K15, model)


def amazon_google(division: int, model: EmbeddingInfo):
    return get_config(f"data/amazon-google/", division, DataSize.K15, model)


def dblp_acm(division: int, model: EmbeddingInfo):
    return get_config(f"data/dblp-acm/", division, DataSize.K15, model)


def dblp_scholar(division: int, model: EmbeddingInfo):
    return get_config(f"data/dblp-scholar/", division, DataSize.K100, model)


def scads(source1: str, source2: str, division: int, model: EmbeddingInfo):
    return get_config(
        f"data/ScadsMB/{source1}-{source2}/", division, DataSize.K15, model
    )


def d_w(version: int, division: int, size: DataSize, model: EmbeddingInfo):
    return get_config(
        f"../../datasets/D_W_{size.value}K_V{version}/", division, size, model
    )


def d_y(version: int, division: int, size: DataSize, model: EmbeddingInfo):
    return get_config(
        f"../../datasets/D_Y_{size.value}K_V{version}/", division, size, model
    )


def en_de(version: int, division: int, size: DataSize, model: EmbeddingInfo):
    return get_config(
        f"../../datasets/EN_DE_{size.value}K_V{version}/", division, size, model
    )


def en_fr(version: int, division: int, size: DataSize, model: EmbeddingInfo):
    return get_config(
        f"../../datasets/EN_FR_{size.value}K_V{version}/", division, size, model
    )


def get_config(path: str, division: int, size: DataSize, emb_info: EmbeddingInfo):
    dataset = OpenEaDataset(
        path,
        f"721_5fold/{division}/",
        f"../OpenEA/run/args/{type(emb_info.model).__name__.lower()}_args_{size.value}K.json",
    )
    emb_info.model.args.output = f"output/results/{dataset.name().replace('/', '-')}/"
    emb_info.model.set_args(emb_info.model.args)
    return dataset, emb_info


def configs():
    # yield from (d_w(1, i, DataSize.K15, boot_ea()) for i in range(1, 6))
    # yield from (d_w(2, i, DataSize.K15, boot_ea()) for i in range(1, 6))
    # yield from (d_w(1, i, DataSize.K15, rdgcn()) for i in range(1, 6))
    # yield from (d_w(2, i, DataSize.K15, rdgcn()) for i in range(1, 6))
    # yield from (d_w(1, i, DataSize.K15, multi_ke()) for i in range(1, 6))
    # yield from (d_w(2, i, DataSize.K15, multi_ke()) for i in range(1, 6))
    # yield from (d_y(1, i, DataSize.K15, boot_ea()) for i in range(1, 6))
    # yield from (d_y(2, i, DataSize.K15, boot_ea()) for i in range(1, 6))
    # yield from (d_y(1, i, DataSize.K15, rdgcn()) for i in range(1, 6))
    # yield from (d_y(2, i, DataSize.K15, rdgcn()) for i in range(1, 6))
    # yield from (d_y(1, i, DataSize.K15, multi_ke()) for i in range(1, 6))
    # yield from (d_y(2, i, DataSize.K15, multi_ke()) for i in range(1, 6))
    # yield from (en_de(1, i, DataSize.K15, boot_ea()) for i in range(1, 6))
    # yield from (en_de(2, i, DataSize.K15, boot_ea()) for i in range(1, 6))
    # yield from (en_de(1, i, DataSize.K15, rdgcn()) for i in range(1, 6))
    # yield from (en_de(2, i, DataSize.K15, rdgcn()) for i in range(1, 6))
#yield from (en_de(1, i, DataSize.K15, multi_ke()) for i in range(3, 6))
    # yield from (en_de(1, i, DataSize.K15, multi_ke()) for i in range(1, 6))
    # yield from (en_de(2, i, DataSize.K15, multi_ke()) for i in range(1, 6))
    # yield from (en_fr(1, i, DataSize.K15, boot_ea()) for i in range(1, 6))
    # yield from (en_fr(2, i, DataSize.K15, boot_ea()) for i in range(1, 6))
    # yield from (en_fr(1, i, DataSize.K15, rdgcn()) for i in range(1, 6))
    # yield from (en_fr(2, i, DataSize.K15, rdgcn()) for i in range(1, 6))
    # yield from (en_fr(1, i, DataSize.K15, multi_ke()) for i in range(1, 6))
    # yield from (en_fr(2, i, DataSize.K15, multi_ke()) for i in range(1, 6))
    # yield from (d_w(1, i, DataSize.K100, boot_ea()) for i in range(1, 6))
    # yield from (d_w(2, i, DataSize.K100, boot_ea()) for i in range(1, 6))
    # yield from (d_w(1, i, DataSize.K100, rdgcn()) for i in range(1, 6))
    # yield from (d_w(2, i, DataSize.K100, rdgcn()) for i in range(1, 6))
    # yield from (d_w(1, i, DataSize.K100, multi_ke()) for i in range(1, 6))
    # yield from (d_w(2, i, DataSize.K100, multi_ke()) for i in range(1, 6))
    # yield from (d_y(1, i, DataSize.K100, boot_ea()) for i in range(1, 6))
    # yield from (d_y(2, i, DataSize.K100, boot_ea()) for i in range(1, 6))
    # yield from (d_y(1, i, DataSize.K100, rdgcn()) for i in range(1, 6))
    # yield from (d_y(2, i, DataSize.K100, rdgcn()) for i in range(1, 6))
    # yield from (d_y(1, i, DataSize.K100, multi_ke()) for i in range(1, 6))
    # yield from (d_y(2, i, DataSize.K100, multi_ke()) for i in range(1, 6))
    # yield from (en_de(1, i, DataSize.K100, boot_ea()) for i in range(1, 6))
    # yield from (en_de(2, i, DataSize.K100, boot_ea()) for i in range(1, 6))
    # yield from (en_de(1, i, DataSize.K100, rdgcn()) for i in range(1, 6))
    # yield from (en_de(2, i, DataSize.K100, rdgcn()) for i in range(1, 6))
    # yield from (en_de(1, i, DataSize.K100, multi_ke()) for i in range(1, 6))
    # yield from (en_de(2, i, DataSize.K100, multi_ke()) for i in range(1, 6))
    # yield from (en_fr(1, i, DataSize.K100, boot_ea()) for i in range(1, 6))
    # yield from (en_fr(2, i, DataSize.K100, boot_ea()) for i in range(1, 6))
    # yield from (en_fr(1, i, DataSize.K100, rdgcn()) for i in range(1, 6))
    # yield from (en_fr(2, i, DataSize.K100, rdgcn()) for i in range(1, 6))
    # yield from (en_fr(1, i, DataSize.K100, multi_ke()) for i in range(1, 6))
    # yield from (en_fr(2, i, DataSize.K100, multi_ke()) for i in range(1, 6))
    yield from (scads("imdb", "tmdb", i, boot_ea()) for i in range(1, 6))
    yield from (scads("imdb", "tmdb", i, rdgcn()) for i in range(1, 6))
    yield from (scads("imdb", "tmdb", i, multi_ke()) for i in range(1, 6))
    yield from (scads("imdb", "tvdb", i, boot_ea()) for i in range(1, 6))
    yield from (scads("imdb", "tvdb", i, rdgcn()) for i in range(1, 6))
    yield from (scads("imdb", "tvdb", i, multi_ke()) for i in range(1, 6))
    yield from (scads("tmdb", "tvdb", i, boot_ea()) for i in range(1, 6))
    yield from (scads("tmdb", "tvdb", i, rdgcn()) for i in range(1, 6))
    yield from (scads("tmdb", "tvdb", i, multi_ke()) for i in range(1, 6))
    yield from (amazon_google(i, boot_ea()) for i in range(1, 6))
    yield from (amazon_google(i, rdgcn()) for i in range(1, 6))
    yield from (amazon_google(i, multi_ke()) for i in range(1, 6))
    yield from (abt_buy(i, boot_ea()) for i in range(1, 6))
    yield from (abt_buy(i, rdgcn()) for i in range(1, 6))
    yield from (abt_buy(i, multi_ke()) for i in range(1, 6))
    yield from (dblp_acm(i, boot_ea()) for i in range(1, 6))
    yield from (dblp_acm(i, rdgcn()) for i in range(1, 6))
    yield from (dblp_acm(i, multi_ke()) for i in range(1, 6))
    yield from (dblp_scholar(i, boot_ea()) for i in range(1, 6))
    yield from (dblp_scholar(i, rdgcn()) for i in range(1, 6))
    yield from (dblp_scholar(i, multi_ke()) for i in range(1, 6))
