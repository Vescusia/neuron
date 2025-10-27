from time import time

import numpy as np


class ReqTimeCalc:
    def __init__(self, wait_time: int):
        """
        :param wait_time: the last wait time.
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
        Estimate the next request time for the player.
        :param satisfaction: [0; 1]; the ratio of satisfaction (new matches in match history / total matches requested)
        :return: tuple (next_request_time, wait_time)
        """
        if satisfaction == 1:
            # quickly reduce waiting time if >= 100 games have been played
            new_wait_time = self.wait_time
            new_wait_time /= 2
            new_wait_time = max(new_wait_time, 86400 * 5)
        else:
            new_wait_time = self.wait_time
            # new_wait_time += np.sqrt(self.wait_time) * (1 - satisfaction)
            # new_wait_time += np.power(self.wait_time, (1/1.1)) * (1. - satisfaction)

            # combine multiple functions to closely approximate the players play rate
            if satisfaction > .75:
                new_wait_time += np.power(1. - satisfaction, 3) * self.initial()[1]
            elif satisfaction > .33:
                new_wait_time += self.wait_time ** 0.981 * (1. - satisfaction)
            else:
                new_wait_time += self.wait_time ** 0.9101 * (1. - satisfaction)

        # calculate the next request time
        next_request_time = time() + new_wait_time

        return int(next_request_time), int(new_wait_time)


if __name__ == "__main__":
    def main():
        from matplotlib import pyplot as plt
        from matplotlib.colors import TABLEAU_COLORS

        rng = np.random.RandomState(int(time()))

        n = 100

        def generate(num_steps: int, games_per_day: float):
            _, wait_time = ReqTimeCalc.initial()
            times = [wait_time]
            satisfactions = [0.]

            for i in range(num_steps - 1):
                satis = rng.normal(games_per_day, .2) * wait_time / 86400 / 100
                satisfactions.append(satis)

                satis = np.clip(satis, 0., 1.)
                _, wait_time = ReqTimeCalc(wait_time).step(satis)
                times.append(wait_time)

            return np.array(times) / 60 / 60 / 24, satisfactions

        fig, ax = plt.subplots()
        fig.set_size_inches(12, 9)

        for games_per_day, color in zip([15, 5, 1, 4/7, 1/7], TABLEAU_COLORS.keys()):
            wait_times, satisfactions = generate(n, games_per_day)
            ax.plot(np.arange(n), wait_times, label=f"{round(games_per_day, 2)} games/day ({np.mean(satisfactions):.2%} avg satis)", color=color)
            ax.plot(np.arange(n), np.full(n, 100 / games_per_day), color=color)

        ax.legend()
        ax.set_ylabel("Estimated Wait Time (days)")
        ax.set_xlabel("Steps")
        ax.set_title(f"Estimated Wait Time (ReqTimeCalc {n} steps) until 100 games have been played")

        fig.show()

    main()
