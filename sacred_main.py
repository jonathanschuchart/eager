from sacred import Experiment as SacredExperiment
from sacred.observers import MongoObserver

from dataset.dataset import DataSize
import main

import numbers

from utils.argument_parser import resolve_data_path

ex = SacredExperiment()
ex.observers.append(MongoObserver())


@ex.config
def config():
    emb_model = "boot_ea"  # either "boot_ea", "multi_ke" or "rdgcn"
    size = DataSize.K15.value
    data_name = "D_W"
    fold = 1  # 1...5
    version = 1  # 1 or 2
    classifier = "random forest 500"  # either "random forest 500" or "MLP"
    ptv_name = "SimAndEmb"
    data_path = resolve_data_path(data_name, size, version)


@ex.capture
def run_single(
    emb_model: str,
    data_path: str,
    fold: int,
    size: int,
    classifier_name: str,
    ptv_name: str,
):
    dataset, emb_info, output_folder = main.resolve_names(
        emb_model, data_path, fold, size
    )
    results, artifacts = main.run_single(
        dataset, emb_info, output_folder, classifier_name, ptv_name
    )

    for file in artifacts:
        ex.add_artifact(file)

    for k, v in results.items():
        if isinstance(v, numbers.Number):
            ex.log_scalar(k.replace("-", "."), float(v))

    return results


@ex.automain
def main():
    return run_single()
