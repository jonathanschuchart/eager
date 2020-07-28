import argparse
import sys
import time

from openea.modules.args.args_hander import check_args, load_args
from openea.modules.load.kgs import read_kgs_from_folder
from openea.models.trans import TransD
from openea.models.trans import TransE
from openea.models.trans import TransH
from openea.models.trans import TransR
from openea.models.semantic import DistMult
from openea.models.semantic import HolE
from openea.models.semantic import SimplE
from openea.models.semantic import RotatE
from openea.models.neural import ConvE
from openea.models.neural import ProjE
from openea.approaches import AlignE
from openea.approaches import BootEA
from openea.approaches import JAPE
from openea.approaches import Attr2Vec
from openea.approaches import MTransE
from openea.approaches import IPTransE
from openea.approaches import GCN_Align
from openea.approaches import AttrE
from openea.approaches import IMUSE
from openea.approaches import SEA
from openea.approaches import MultiKE
from openea.approaches import RSN4EA
from openea.approaches import GMNN
from openea.approaches import KDCoE
from openea.approaches import RDGCN
from openea.approaches import BootEA_RotatE
from openea.approaches import BootEA_TransH
from openea.models.basic_model import BasicModel


class ModelFamily(object):
    BasicModel = BasicModel

    TransE = TransE
    TransD = TransD
    TransH = TransH
    TransR = TransR

    DistMult = DistMult
    HolE = HolE
    SimplE = SimplE
    RotatE = RotatE

    ProjE = ProjE
    ConvE = ConvE

    MTransE = MTransE
    IPTransE = IPTransE
    Attr2Vec = Attr2Vec
    JAPE = JAPE
    AlignE = AlignE
    BootEA = BootEA
    GCN_Align = GCN_Align
    GMNN = GMNN
    KDCoE = KDCoE

    AttrE = AttrE
    IMUSE = IMUSE
    SEA = SEA
    MultiKE = MultiKE
    RSN4EA = RSN4EA
    RDGCN = RDGCN
    BootEA_RotatE = BootEA_RotatE
    BootEA_TransH = BootEA_TransH


def get_model(model_name):
    return getattr(ModelFamily, model_name)


def main_for_args(arg_path, dataset, division):
    t = time.time()
    args = load_args(arg_path)
    args.training_data = args.training_data + dataset + "/"
    args.dataset_division = division
    print(args.embedding_module)
    print(args)
    remove_unlinked = False
    if args.embedding_module == "RSN4EA":
        remove_unlinked = True
    kgs = read_kgs_from_folder(
        args.training_data,
        args.dataset_division,
        args.alignment_module,
        args.ordered,
        remove_unlinked=remove_unlinked,
    )
    import tensorflow as tf

    tf.keras.backend.clear_session()
    model = get_model(args.embedding_module)()
    model.set_args(args)
    model.set_kgs(kgs)
    model.init()
    model.run()
    model.test()
    model.save()
    print("Total run time = {:.3f} s.".format(time.time() - t))


def main():
    main_for_args(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    for dataset in ["dblp-acm"]:
        for i in range(5, 6):
            main_for_args(
                "../OpenEA/run/args/bootea_args_15K.json", dataset, f"721_5fold/{i}/"
            )
