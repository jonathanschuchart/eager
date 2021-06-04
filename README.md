# EAGER

This repo can be used to replicate the experiments described in our [paper on
"EAGER - Embedding Assisted Knowledge Graph Entity Resolution"](https://dbs.uni-leipzig.de/file/EAGERpreprint.pdf)

## Installation
The easiest way is to use [poetry](https://python-poetry.org/docs/):
```bash
git clone https://github.com/jonathanschuchart/eager 
# OpenEA is needed if you want to calculate the embeddings yourself
git clone https://github.com/nju-websoft/OpenEA ../OpenEA
# specifiy a python3.7 version because OpenEA uses an old tensorflow version
python env use python3.7
poetry install
```

For running you have to modify your PYTHONPATH:
If you are using bash/zsh:
```bash
export PYTHONPATH=$PYTHONPATH:src
```
If you are using fish:
```fish
set -x PYTHONPATH $PYTHONPATH src

```
## Prerequisites
You will need a few things downloaded and stored in the right places before the experiments
can be run properly:

- datasets:
  The [OpenEA datasets](https://www.dropbox.com/s/nzjxbam47f9yk3d/OpenEA_dataset_v1.1.zip?dl=0),
  [movie datasets](https://github.com/ScaDS/MovieGraphBenchmark) and
  [CSV datasets](https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution)
  need to be stored next to this repo in a folder called `datasets`.
  Experiments in the paper have been run with the OpenEA datasets version 1.1,
  though version 2.0 should work just as well (this has not been tested).
  
- word2vec:
  The [word2vec embeddings](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip)
  need to be unzipped and stored in the datasets folder.
  
- data split:
  In order to run the experiments on the shallow datasets, run `split_datasets.py`.
  This will create a 5-fold 7-2-1 split of the non-OpenEA datasets in accordance with the OpenEA format.
  
## How to run
Simply running `python main.py` will replicate all experiments mentioned in the paper.
Be aware that there are a lot of experiments that will take a lot of time.

The respective embeddings needed for the experiments can be computed as needed but
can also be pre-computed with OpenEA.
The pre-computed embeddings should live in a folder called `output/results`
next to this repo (this is the default from OpenEA).

## Utilities
A few utilities for collecting the results are `src/utils/collect_predictions.py` and
`src/utils/combine_csv.py`. These can be used to collect all prediction files into a zip archive
and combining all result files (containing f-measure, recall, precision) into one large csv file.

## Experimental Setups
In order to define a specific set of experiments, a number of parameters can be defined
via the command line.
The list of parameters is available via `python main.py -h`. Each parameter is optional and
defaults to a list of available values. For example, if you'd only like to run only
with the embedding models `boot_ea` and `multi_ke` you can specify `--emb_models boot_ea multi_ke`.
