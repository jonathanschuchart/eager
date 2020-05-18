def get_attr_dict(path: str) -> dict:
    attr_dict = dict()
    with open(path, "r") as f:
        for line in f:
            subj, pred, obj = line.strip().split("\t")
            attr_dict[subj] = (pred, obj)
    return attr_dict


def get_attr_dict_with_emb_ids(
    path: str, kg1_entities: dict, kg2_entities: dict
) -> dict:
    attr_dict = dict()
    with open(path, "r") as f:
        for line in f:
            subj, pred, obj = line.strip().split("\t")
            if subj in kg1_entities:
                attr_dict[kg1_entities[subj]] = (pred, obj)
            elif subj in kg2_entities:
                attr_dict[kg2_entities[subj]] = (pred, obj)
    return attr_dict
