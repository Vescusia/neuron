import datetime
import signal
import time
from threading import Thread
from queue import Queue
from pathlib import Path
import os

import cassiopeia as cass
import lmdb

from continent_db import ContinentDB
import match_scrape_thread
from syn_db import SynergyDB, CodedSynergy


def main(api_key: str, continents: list[cass.data.Continent], match_db_path: str, sum_db_path: str, syn_db_path: str, matchup_db_path: str, matches_path: Path) -> None:
    print(api_key, continents, match_db_path, sum_db_path)
    cass.print_calls(False)

    # open database environments
    os.makedirs(match_db_path, exist_ok=True)
    match_env = lmdb.open(match_db_path, max_dbs=len(continents))
    print("MatchDB stats:", match_env.info())
    os.makedirs(syn_db_path, exist_ok=True)
    sum_env = lmdb.open(sum_db_path, max_dbs=len(continents))
    print("SummonerDB stats:", sum_env.info())
    os.makedirs(syn_db_path, exist_ok=True)
    syn_db = SynergyDB(syn_db_path)
    print("SynergyDB stats:", syn_db.env.info())
    os.makedirs(match_db_path, exist_ok=True)
    matchup_db = SynergyDB(matchup_db_path)
    print("MatchupDB stats:", matchup_db.env.info())

    # create Continent specific Databases
    cont_dbs: list[tuple[ContinentDB, ContinentDB]] = []  # (MatchDB, SumDB)
    for continent in continents:
        cont_dbs.append((ContinentDB(match_env, continent), ContinentDB(sum_env, continent)))

    # start thread for each continent
    stop_q, state_q = Queue(), Queue()
    futures = []
    for dbs in cont_dbs:
        t = Thread(target=match_scrape_thread.scrape_continent, args=(stop_q, state_q, *dbs, syn_db, matchup_db, matches_path))
        futures.append(t)
        t.start()

    # start state printer
    Thread(target=state_printer, args=(state_q,), daemon=True).start()

    # wait for stop signal from user
    signal.signal(signal.SIGINT, lambda x, y: [stop_q.put(None), print("\nShutting down, please stand by...")])
    for future in futures:
        # has to be done this way to receive signals ¯\_(ツ)_/¯
        while future.is_alive():
            future.join(1)


# Print the current state of the continent scrapers
# Just prints out the amount of total explored matches across threads
def state_printer(state_q: Queue[int]) -> None:
    total = 0
    start = time.time()
    while True:
        total += state_q.get()
        print(f"[{datetime.datetime.now().strftime('%H:%M-%d.%m.%y')}] {total} Matches explored. ({total / (time.time() - start):02f}/s)")
