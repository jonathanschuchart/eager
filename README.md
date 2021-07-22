# EAGER

This repo can be used to replicate the experiments described in our [paper on
"EAGER - Embedding Assisted Knowledge Graph Entity Resolution"](https://dbs.uni-leipzig.de/file/EAGERpreprint.pdf)

## Prerequisites
You will need a few things downloaded and stored in the right places before the experiments
can be run properly:

- OpenEA:
  A copy of the [OpenEA library](https://github.com/nju-websoft/OpenEA) needs to be stored next to this repo

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

## Citing
If you use our work please use the following citation:
```
@inproceedings{EAGERKGCW2021,
  author    = {Daniel Obraczka and
               Jonathan Schuchart and
               Erhard Rahm},
  editor    = {David Chaves-Fraga and
               Anastasia Dimou and
               Pieter Heyvaert and
               Freddy Priyatna and
               Juan Sequeda},
  title     = {Embedding-Assisted Entity Resolution for Knowledge Graphs},
  booktitle = {Proceedings of the 2nd International Workshop on Knowledge Graph Construction co-located with 18th Extended Semantic Web Conference (ESWC 2021), Online, June 5, 2021},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {2873},
  publisher = {CEUR-WS.org},
  year      = {2021},
  url       = {http://ceur-ws.org/Vol-2873/paper8.pdf},
}
```
