from openea.approaches import GCN_Align, BootEA, RDGCN, MultiKE
from openea.models.basic_model import BasicModel
from openea.modules.args.args_hander import load_args

from dataset.csv_dataset import CsvDataset, CsvType
from dataset.openea_dataset import OpenEaDataset
from dataset.scads_dataset import ScadsDataset


# Algorithms:
def gcn_align_15():
    model = GCN_Align()
    model.set_args(load_args("../OpenEA/run/args/gcnalign_args_15K.json"))
    return model, "gcn_align_15", lambda m: m.vec_se


def boot_ea_15():
    model = BootEA()
    model.set_args(load_args("../OpenEA/run/args/bootea_args_15K.json"))
    return model, "boot_ea_15", lambda m: m.ent_embeds.eval(session=m.session)


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
def amazon_google(model: tuple):
    dataset = CsvDataset(
        CsvType.products,
        "data/amazon-google/Amazon.csv",
        "data/amazon-google/GoogleProducts.csv",
        "data/amazon-google/Amzon_GoogleProducts_perfectMapping.csv",
        model[0],
    )
    return lambda: (dataset, model)


def dblp_acm(model: tuple):
    dataset = CsvDataset(
        CsvType.articles,
        "data/dblp-acm/DBLP2.csv",
        "data/dblp-acm/ACM.csv",
        "data/dblp-acm/DBLP-ACM_perfectMapping.csv",
        model[0],
    )
    return lambda: (dataset, model)


def d_w_15k(version: int, division: int, model_name: str, model: tuple):
    return lambda: (
        OpenEaDataset(
            f"../../datasets/D_W_15K_V{version}/",
            f"721_5fold/{division}/",
            f"../OpenEA/run/args/{model_name}_args_15K.json",
        ),
        model,
    )


def d_y_15k(version: int, division: int, model_name: str, model: tuple):
    return lambda: (
        OpenEaDataset(
            f"../../datasets/D_Y_15K_V{version}/",
            f"721_5fold/{division}/",
            f"../OpenEA/run/args/{model_name}_args_15K.json",
        ),
        model,
    )


def en_de_15k(version: int, division: int, model_name: str, model: tuple):
    return lambda: (
        OpenEaDataset(
            f"../../datasets/EN_DE_15K_V{version}/",
            f"721_5fold/{division}/",
            f"../OpenEA/run/args/{model_name}_args_15K.json",
        ),
        model,
    )


def en_fr_15k(version: int, division: int, model_name: str, model: tuple):
    return lambda: (
        OpenEaDataset(
            f"../../datasets/EN_FR_15K_V{version}/",
            f"721_5fold/{division}/",
            f"../OpenEA/run/args/{model_name}_args_15K.json",
        ),
        model,
    )


def scads(source1: str, source2: str, model: tuple):
    return lambda: (ScadsDataset("data/ScadsMB/100", source1, source2, model[0]), model)


configs = [
    *[d_w_15k(1, i, "bootea", boot_ea_15()) for i in range(1, 6)],
    *[d_w_15k(2, i, "bootea", boot_ea_15()) for i in range(1, 6)],
    *[d_w_15k(1, i, "rdgcn", rdgcn()) for i in range(1, 6)],
    *[d_w_15k(2, i, "rdgcn", rdgcn()) for i in range(1, 6)],
    *[d_w_15k(1, i, "multike", multi_ke()) for i in range(1, 6)],
    *[d_w_15k(2, i, "multike", multi_ke()) for i in range(1, 6)],
    *[d_y_15k(1, i, "bootea", boot_ea_15()) for i in range(1, 6)],
    *[d_y_15k(2, i, "bootea", boot_ea_15()) for i in range(1, 6)],
    *[d_y_15k(1, i, "rdgcn", rdgcn()) for i in range(1, 6)],
    *[d_y_15k(2, i, "rdgcn", rdgcn()) for i in range(1, 6)],
    *[d_y_15k(1, i, "multike", multi_ke()) for i in range(1, 6)],
    *[d_y_15k(2, i, "multike", multi_ke()) for i in range(1, 6)],
    *[en_de_15k(1, i, "bootea", boot_ea_15()) for i in range(1, 6)],
    *[en_de_15k(2, i, "bootea", boot_ea_15()) for i in range(1, 6)],
    *[en_de_15k(1, i, "rdgcn", rdgcn()) for i in range(1, 6)],
    *[en_de_15k(2, i, "rdgcn", rdgcn()) for i in range(1, 6)],
    *[en_de_15k(1, i, "multike", multi_ke()) for i in range(1, 6)],
    *[en_de_15k(2, i, "multike", multi_ke()) for i in range(1, 6)],
    *[en_fr_15k(1, i, "bootea", boot_ea_15()) for i in range(1, 6)],
    *[en_fr_15k(2, i, "bootea", boot_ea_15()) for i in range(1, 6)],
    *[en_fr_15k(1, i, "rdgcn", rdgcn()) for i in range(1, 6)],
    *[en_fr_15k(2, i, "rdgcn", rdgcn()) for i in range(1, 6)],
    *[en_fr_15k(1, i, "multike", multi_ke()) for i in range(1, 6)],
    *[en_fr_15k(2, i, "multike", multi_ke()) for i in range(1, 6)],
    scads("imdb", "tmdb", gcn_align_15()),
    # scads("imdb", "tmdb", boot_ea_15()),
    scads("imdb", "tmdb", rdgcn()),
    # scads("imdb", "tmdb", multi_ke()),
    scads("imdb", "tvdb", gcn_align_15()),
    # scads("imdb", "tvdb", boot_ea_15()),
    scads("imdb", "tvdb", rdgcn()),
    # scads("imdb", "tvdb", multi_ke()),
    scads("tmdb", "tvdb", gcn_align_15()),
    # scads("tmdb", "tvdb", boot_ea_15()),
    scads("tmdb", "tvdb", rdgcn()),
    # scads("tmdb", "tvdb", multi_ke()),
    amazon_google(gcn_align_15()),
    # amazon_google(boot_ea_15()),
    amazon_google(rdgcn()),
    # amazon_google(multi_ke()),
    dblp_acm(gcn_align_15()),
    # dblp_acm(boot_ea_15()),
    dblp_acm(rdgcn()),
    # dblp_acm(multi_ke()),
]
