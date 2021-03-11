from os import path

from sentence_transformers import SentenceTransformer

from attribute_features import _remove_type
from dataset.dataset import Dataset
from similarity_measures import bert_embed


def create_bert_embeddings(dataset: Dataset, output_folder: str):
    if path.exists(path.join(output_folder, "bert_embeds_1")) and path.exists(
        path.join(output_folder, "bert_embeds_2")
    ):
        return

    print(f"Creating Bert embeddings for {dataset.name()} in {output_folder}")
    kgs = dataset.kgs()
    bert_key = "distilbert-multilingual-nli-stsb-quora-ranking"
    bert_model = SentenceTransformer(bert_key)
    kg1_dict = kgs.kg1.av_dict
    bert_embeds_1 = []
    for e1, e1_attrs in kg1_dict.items():
        v1 = " ".join(_remove_type(v) for _, v in sorted(e1_attrs))
        bert_embeds_1.append((e1, v1, bert_embed(bert_model, v1)))

    kg2_dict = kgs.kg2.av_dict
    bert_embeds_2 = []
    for e2, e2_attrs in kg2_dict.items():
        v2 = " ".join(_remove_type(v) for _, v in sorted(e2_attrs))
        bert_embeds_2.append((e2, v2, bert_embed(bert_model, v2)))

    np.save(path.join(output_folder, "bert_embeds_1"), bert_embeds_1)
    np.save(path.join(output_folder, "bert_embeds_2"), bert_embeds_2)
