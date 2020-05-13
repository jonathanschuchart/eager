class NumberDistance(object):
    """
    Calculates the absolute distance of number strings
    """

    def get_distance(self, attr1: str, attr2: str):
        try:
            return abs(int(attr1) - int(attr2))
        except Exception as e:
            try:
                return abs(float(attr1) - float(attr2))
            except Exception as e:
                raise e
