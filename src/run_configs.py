from openea.approaches import GCN_Align, BootEA, RDGCN, MultiKE
from openea.modules.args.args_hander import load_args

from dataset.dataset import DataSize
from dataset.openea_dataset import OpenEaDataset


# Algorithms:
def gcn_align_15():
    model = GCN_Align()
    model.set_args(load_args("../OpenEA/run/args/gcn_align_args_15K.json"))
    return model, "gcn_align", lambda m: m.vec_se


def boot_ea():
    model = BootEA()
    model.set_args(load_args("../OpenEA/run/args/bootea_args_15K.json"))
    return model, "boot_ea", lambda m: m.ent_embeds.eval(session=m.session)


def rdgcn():
    model = RDGCN()
    model.set_args(load_args("../OpenEA/run/args/rdgcn_args_15K.json"))
    return model, "rdgcn", lambda m: m.sess.run(m.output)


def multi_ke():
    model = MultiKE()
    model.set_args(load_args("../OpenEA/run/args/multike_args_15K.json"))
    return model, "multi_ke", lambda m: m.ent_embeds.eval(session=m.session)


def gcn_align_100():
    model = GCN_Align()
    args = load_args("../OpenEA/run/args/gcnalign_args_100K.json")
    model.set_args(args)
    return model, "gcn_align_100", lambda m: m.vec_se


# Datasets:
def abt_buy(division: int, model: tuple):
    return get_config(f"data/abt-buy/", division, model)


def amazon_google(division: int, model: tuple):
    return get_config(f"data/amazon-google/", division, model)


def dblp_acm(division: int, model: tuple):
    return get_config(f"data/dblp-acm/", division, model)


def dblp_scholar(division: int, model: tuple):
    return get_config(f"data/dblp-scholar/", division, model, DataSize.K100)


def scads(source1: str, source2: str, division: int, model: tuple):
    return get_config(f"data/ScadsMB/{source1}-{source2}/", division, model)


def d_w_15k(version: int, division: int, model: tuple):
    return get_config(f"../../datasets/D_W_15K_V{version}/", division, model)


def d_y_15k(version: int, division: int, model: tuple):
    return get_config(f"../../datasets/D_Y_15K_V{version}/", division, model)


def en_de_15k(version: int, division: int, model: tuple):
    return get_config(f"../../datasets/EN_DE_15K_V{version}/", division, model)


def en_fr_15k(version: int, division: int, model: tuple):
    return get_config(f"../../datasets/EN_FR_15K_V{version}/", division, model)


def get_config(path: str, division: int, model: tuple, size: DataSize = DataSize.K15):
    return lambda: (
        OpenEaDataset(
            path,
            f"721_5fold/{division}/",
            f"../OpenEA/run/args/{type(model[0]).__name__.lower()}_args_{size.value}K.json",
        ),
        model,
    )


configs = [
    *[d_w_15k(1, i, boot_ea()) for i in range(1, 6)],
    *[d_w_15k(2, i, boot_ea()) for i in range(1, 6)],
    *[d_w_15k(1, i, rdgcn()) for i in range(1, 6)],
    *[d_w_15k(2, i, rdgcn()) for i in range(1, 6)],
    *[d_w_15k(1, i, multi_ke()) for i in range(1, 6)],
    *[d_w_15k(2, i, multi_ke()) for i in range(1, 6)],
    *[d_y_15k(1, i, boot_ea()) for i in range(1, 6)],
    *[d_y_15k(2, i, boot_ea()) for i in range(1, 6)],
    *[d_y_15k(1, i, rdgcn()) for i in range(1, 6)],
    *[d_y_15k(2, i, rdgcn()) for i in range(1, 6)],
    *[d_y_15k(1, i, multi_ke()) for i in range(1, 6)],
    *[d_y_15k(2, i, multi_ke()) for i in range(1, 6)],
    *[en_de_15k(1, i, boot_ea()) for i in range(1, 6)],
    *[en_de_15k(2, i, boot_ea()) for i in range(1, 6)],
    *[en_de_15k(1, i, rdgcn()) for i in range(1, 6)],
    *[en_de_15k(2, i, rdgcn()) for i in range(1, 6)],
    *[en_de_15k(1, i, multi_ke()) for i in range(1, 6)],
    *[en_de_15k(2, i, multi_ke()) for i in range(1, 6)],
    *[en_fr_15k(1, i, boot_ea()) for i in range(1, 6)],
    *[en_fr_15k(2, i, boot_ea()) for i in range(1, 6)],
    *[en_fr_15k(1, i, rdgcn()) for i in range(1, 6)],
    *[en_fr_15k(2, i, rdgcn()) for i in range(1, 6)],
    *[en_fr_15k(1, i, multi_ke()) for i in range(1, 6)],
    *[en_fr_15k(2, i, multi_ke()) for i in range(1, 6)],
    *[scads("imdb", "tmdb", i, gcn_align_15()) for i in range(1, 6)],
    *[scads("imdb", "tmdb", i, boot_ea()) for i in range(1, 6)],
    *[scads("imdb", "tmdb", i, rdgcn()) for i in range(1, 6)],
    *[scads("imdb", "tmdb", i, multi_ke()) for i in range(1, 6)],
    *[scads("imdb", "tvdb", i, gcn_align_15()) for i in range(1, 6)],
    *[scads("imdb", "tvdb", i, boot_ea()) for i in range(1, 6)],
    *[scads("imdb", "tvdb", i, rdgcn()) for i in range(1, 6)],
    *[scads("imdb", "tvdb", i, multi_ke()) for i in range(1, 6)],
    *[scads("tmdb", "tvdb", i, gcn_align_15()) for i in range(1, 6)],
    *[scads("tmdb", "tvdb", i, boot_ea()) for i in range(1, 6)],
    *[scads("tmdb", "tvdb", i, rdgcn()) for i in range(1, 6)],
    *[scads("tmdb", "tvdb", i, multi_ke()) for i in range(1, 6)],
    *[amazon_google(i, boot_ea()) for i in range(1, 6)],
    *[amazon_google(i, rdgcn()) for i in range(1, 6)],
    # *[amazon_google(i, multi_ke()) for i in range(1, 6)],
    *[abt_buy(i, boot_ea()) for i in range(1, 6)],
    *[abt_buy(i, rdgcn()) for i in range(1, 6)],
    # *[abt_buy(i, multi_ke()) for i in range(1, 6)],
    *[dblp_acm(i, boot_ea()) for i in range(1, 6)],
    *[dblp_acm(i, rdgcn()) for i in range(1, 6)],
    # *[dblp_acm(i, multi_ke()) for i in range(1, 6)],
    *[dblp_scholar(i, boot_ea()) for i in range(1, 6)],
    *[dblp_scholar(i, rdgcn()) for i in range(1, 6)],
    *[dblp_scholar(i, multi_ke()) for i in range(1, 6)],
]
