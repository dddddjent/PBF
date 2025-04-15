# a timer

import time


class Timer:
    def __init__(self):
        self.start = time.time()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start
