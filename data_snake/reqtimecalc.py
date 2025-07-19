from time import time

import numpy as np


class ReqTimeCalc:
    def __init__(self, wait_time: int):
        """
        :param wait_time: the initial wait time with which will be calculated.
        """
        self.wait_time = wait_time

    @staticmethod
    def initial() -> tuple[int, int]:
        """
        :return: tuple (next request time, initial wait time) where the initial wait time is 30 days and the summoner may be instantly requested.
        """
        return 0, 86400 * 30

    def step(self, satisfaction: float) -> tuple[int, int]:
        """
        Calculate the next request time and factor.
        :param satisfaction: [0; 1]; the ratio of satisfaction (new matches in match history / total matches requested)
        :return: tuple (next_request_time, new_factor)
        """
        # calculate new factor
        if satisfaction == 1:
            new_wait_time = min(self.wait_time / 2, 86400 * 5)
        else:
            new_wait_time = int(self.wait_time + np.sqrt(self.wait_time) * (1 - satisfaction))

        # calculate the next request time
        next_request_time = int(time()) + new_wait_time

        return next_request_time, new_wait_time
