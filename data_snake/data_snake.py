from datetime import datetime
import signal
from time import time
from threading import Thread
from queue import Queue
from pathlib import Path
import os

import lmdb
import riotwatcher as rw

from continent_db import ContinentDB, MatchDB, SummonerDB
from match_request_thread import crawl_continent
from league_arrow import LolDataset


def main(continents: list[str], match_db_path: str, sum_db_path: str, matches_path: Path, dataset_path: Path,
         lolwatcher: rw.LolWatcher) -> None:
    # open database environments
    os.makedirs(match_db_path, exist_ok=True)
    match_env = lmdb.open(match_db_path, max_dbs=len(continents))
    print("MatchDB stats:", match_env.info())
    os.makedirs(sum_db_path, exist_ok=True)
    sum_env = lmdb.open(sum_db_path, max_dbs=len(continents))
    print("SummonerDB stats:", sum_env.info())

    # open dataset
    dataset = LolDataset(dataset_path, write_interval=3_000)  # write once per hour

    # start thread for each continent
    stop_q, state_q = Queue(), Queue()
    futures = []
    for continent in continents:
        t = Thread(target=crawl_continent, args=(
            stop_q,
            state_q,
            MatchDB(ContinentDB(match_env, continent)),
            SummonerDB(ContinentDB(sum_env, continent)),
            matches_path,
            dataset.open_continent(continent),
            lolwatcher
        ))
        futures.append(t)
        t.start()

    # start state printer
    Thread(target=state_printer, args=(state_q,), daemon=True).start()

    # wait for stop signal from user
    signal.signal(signal.SIGINT,
                  lambda x, y: [[stop_q.put(None) for _ in continents], print("\nShutting down, please stand by...")])
    for future in futures:
        # has to be done this way to receive signals ¯\_(ツ)_/¯
        while future.is_alive():
            future.join(1)


# Print the current state of the continent scrapers
# Just prints out the number of total explored matches across threads
def state_printer(state_q: Queue[int]) -> None:
    total = 0
    start = time()
    while True:
        total += state_q.get()
        print(
            f"[{datetime.now().strftime('%H:%M-%d.%m.%y')}] {total} Matches explored. ({total / (time() - start):02f}/s)")
