class Statistics(object):

    def __init__(self):
        self._bees_in = 0
        self._bees_out = 0
        self._bees_in_overall = 0
        self._bees_out_overall = 0
        self._processed_fames = 0
        self._processed_fames_overlall = 0

    def frame_processed(self):
        self._processed_fames += 1
        self._processed_fames_overlall += 1

    def add_in(self):
        self._bees_in += 1
        self._bees_in_overall += 1

    def add_out(self):
        self._bees_out += 1
        self._bees_out_overall += 1

    def get_count_over(self):
        return (self._bees_in_overall, self._bees_out_overall)

    def get_count(self):
        return (self._bees_in, self._bees_out)


__dh = None


def get_statistics():
    global __dh
    if __dh == None:
        __dh = Statistics()

    return __dh
