from datetime import datetime
import signal
from time import time
from threading import Thread
from queue import Queue
from pathlib import Path
import os

import lmdb
import riotwatcher as rw

from .continent_db import ContinentDB, MatchDB, SummonerDB
from .match_request_thread import crawl_continent
from lib.league_of_parquet import LolDatasetWriter


def gather(continents: list[str], match_db_path: str, sum_db_path: str, matches_path: Path, dataset_path: Path,
           lolwatcher: rw.LolWatcher) -> None:
    # open database environments
    os.makedirs(match_db_path, exist_ok=True)
    match_env = lmdb.open(match_db_path, map_size=1_000_000_000, max_dbs=len(continents))
    os.makedirs(sum_db_path, exist_ok=True)
    sum_env = lmdb.open(sum_db_path, map_size=2_250_000_000, max_dbs=len(continents))

    # print environment state
    print(f"SummonerDB: {match_env.stat()['psize'] * match_env.info()['last_pgno'] / match_env.info()['map_size'] * 100:.1f} % full ({match_env.info()})")
    print(f"SummonerDB: {sum_env.stat()['psize'] * sum_env.info()['last_pgno'] / sum_env.info()['map_size'] * 100:.1f} % full ({sum_env.info()})")

    # open dataset
    dataset = LolDatasetWriter(dataset_path, write_interval=3_000)  # write once per hour

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
                  lambda x, y: [stop_q.put(None), print("\nShutting down, please stand by...")])
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
