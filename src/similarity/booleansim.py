class BooleanSimilarity(object):
    """
    Used to check if boolean strings are the same
    Necessary to provide common api for distance/similarity
    """

    def get_sim_score(self, attr1: str, attr2: str):
        """
        :param attr1: one boolean string
        :param attr2: other boolean string
        :return: 1 if strings are the same boolean value, 0 else
        """
        if attr1.lower().strip() == attr2.lower().strip():
            return 1.0
        return 0.0
