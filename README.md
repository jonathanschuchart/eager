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

## How to run
Simply running `poetry run python main.py` will replicate all experiments mentioned in the paper and download the necessary datasets if needed.
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
