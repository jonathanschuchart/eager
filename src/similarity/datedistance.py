from datetime import date


class DateDistance(object):
    """
    Used to calculate the distance between datestrings
    """

    def _create_date(self, date_string: str):
        if not "-" in date_string:
            # only year
            return date(int(date_string), 1, 1)
        if date_string.count("-") == 2:
            year, month, day = date_string.split("-")
            return date(int(year), int(month), int(day))
        else:
            year, month = date_string.split("-")
            return date(int(year), int(month), 1)

    def get_distance(self, date_string1: str, date_string2: str) -> int:
        """
        Dates will be parsed to datetime.date and the absolute difference in dates is returned
        :param date_string1: one date as string
        :param date_string2: other date string
        :return: difference in days
        """
        date1 = self._create_date(date_string1)
        date2 = self._create_date(date_string2)
        return abs(date1 - date2).days
