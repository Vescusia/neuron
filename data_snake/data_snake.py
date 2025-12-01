from datetime import datetime
import signal
from time import time
from threading import Thread
from queue import Queue
from pathlib import Path
import os

import lmdb
import numpy as np
import riotwatcher as rw

from .continent_db import ContinentDB, MatchDB, SummonerDB
from .continent_crawler import crawl
from lib.league_of_parquet import LolDatasetWriter


def gather(continents: list[str], match_db_path: Path, sum_db_path: Path, matches_path: Path, dataset_path: Path,
           lolwatcher: rw.LolWatcher, match_map_size: int, sum_map_size: int) -> None:
    # open database environments
    os.makedirs(match_db_path, exist_ok=True)
    match_env = lmdb.open(str(match_db_path), map_size=match_map_size, max_dbs=len(continents))
    os.makedirs(sum_db_path, exist_ok=True)
    sum_env = lmdb.open(str(sum_db_path), map_size=sum_map_size, max_dbs=len(continents))

    # print environment state
    print(f"MatchDB: {match_env.stat()['psize'] * match_env.info()['last_pgno'] / match_env.info()['map_size'] * 100:.1f} % full ({match_env.info()})")
    print(f"SummonerDB: {sum_env.stat()['psize'] * sum_env.info()['last_pgno'] / sum_env.info()['map_size'] * 100:.1f} % full ({sum_env.info()})")

    # open dataset
    dataset = LolDatasetWriter(dataset_path, write_interval=3_000)  # write once per hour

    # start thread for each continent
    stop_q, state_q = Queue(), Queue()
    crawlers: dict[str, Thread] = {}
    for continent in continents:
        t = Thread(target=crawl, args=(
            stop_q,
            state_q,
            MatchDB(ContinentDB(match_env, continent)),
            SummonerDB(ContinentDB(sum_env, continent)),
            matches_path,
            dataset.open_continent(continent),
            lolwatcher
        ))
        crawlers[continent] = t
        t.start()

    # start state printer
    Thread(target=state_printer, args=(state_q, crawlers), daemon=True).start()

    # wait for stop signal from user
    signal.signal(signal.SIGINT,
                  lambda x, y: [stop_q.put(None), print("\nShutting down, please stand by...")])
    for future in crawlers.values():
        # has to be done this way to receive signals ¯\_(ツ)_/¯
        while future.is_alive():
            future.join(1)


# Print the current state of the continent crawlers
# Just prints out the number of total explored matches across threads
def state_printer(state_q: Queue, crawlers: dict[str, Thread]) -> None:
    total_matches_explored = 0
    satisfactions = []

    start = time()
    while True:
        # get state from Queue
        # inc_matches_explored: actual number of matches that got requested, explored and analyzed
        # satisfaction: number of new matches / total number of matches in player match history
        inc_matches_explored, satisfaction = state_q.get()

        # increment total variables
        total_matches_explored += inc_matches_explored
        satisfactions.append(satisfaction)

        # check which continent threads are still alive
        alive_crawlers = [continent[0:2] for continent, thread in crawlers.items() if thread.is_alive()]

        # print state
        print(
            f"[{datetime.now().strftime('%H:%M-%d.%m.%y')} | {' '.join(alive_crawlers)}] "
            f"{total_matches_explored:06} matches explored "
            f"({total_matches_explored / (time() - start):.2f}/s), "
            f"satisfaction: {np.average(satisfactions):05.1%} mean with {np.std(satisfactions):05.1%} std"
              )
