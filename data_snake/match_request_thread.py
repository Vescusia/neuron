from pathlib import Path
from time import time, sleep
from queue import Queue
import traceback

import requests
import riotwatcher as rw

import lib
from lib.league_of_parquet import ContinentDatasetWriter
from .continent_db import MatchDB, SummonerDB
from .compressed_json_ball import CompressedJSONBall
from .reqtimecalc import ReqTimeCalc


def crawl_continent(stop_q: Queue[None], state_q: Queue[int], match_db: MatchDB, sum_db: SummonerDB, matches_path: Path, dataset: ContinentDatasetWriter, lolwatcher: rw.LolWatcher) -> None:
    # print database state
    print(f"{sum_db.continent}: {sum_db.count()} Summoners, {match_db.count()} Matches")

    # variable for incremental explored matches,
    # with incremental meaning between updates to state_q
    inc_explored_matches = 0

    # create matches_dir for JSON files
    matches_path = matches_path / match_db.continent / f"{int(time())}.gzip"
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    # open lzma compressed JSON ball
    matches_ball = CompressedJSONBall(matches_path, split_every=36_000)

    while True:
        try:
            # break if we get the signal to stop
            if not stop_q.empty():
                state_q.put(inc_explored_matches)
                break

            # search for unexplored match
            while True:
                # query match database for an unexplored match
                unexplored_match = match_db.unexplored_match()
                if unexplored_match is not None:
                    new_match_id, ranked_score = unexplored_match
                    break
                # if the match database does not contain any unexplored matches anymore
                else:
                    # request match history from an unexplored player
                    explore_player(match_db, sum_db, stop_q, lolwatcher)
                    # update explored matches
                    state_q.put(inc_explored_matches)
                    inc_explored_matches = 0

            # request Match from RiotAPI
            try:
                new_match = lolwatcher.match.by_id(match_db.continent, new_match_id)
            except rw.ApiError as e:
                # check if this match id is invalid
                if e.response.status_code == 404:
                    print(f"\n[ERROR] Match {new_match_id} is invalid, skipping...\n")
                    # mark as explored
                    match_db.set_explored(new_match_id)
                    continue
                else:
                    raise e

            # add match to dataset
            dataset.append(new_match, ranked_score)

            # save match JSON
            matches_ball.append(new_match)

            # add participants to SummonerDB
            puuids = [participant['puuid'] for participant in new_match['info']['participants']]
            request_times = [ReqTimeCalc.initial() for _ in puuids]
            sum_db.put_multi([(puuid, req, wait_time) for puuid, (req, wait_time) in zip(puuids, request_times)])

            # mark Match as explored
            match_db.set_explored(new_match_id)
            inc_explored_matches += 1

        # catch the errors that just sometimes happen with web traffic.
        except (requests.exceptions.HTTPError, requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
            print("\n[ERROR]:")
            traceback.print_exception(e)
            print("\nContinuing...")
            sleep(5)
            continue

    # close the matches ball
    matches_ball.close()

    # write matches between intervals to dataset
    dataset.write_match_list()


def explore_player(match_db: MatchDB, sum_db: SummonerDB, stop_q: Queue[None], lolwatcher: rw.LolWatcher) -> None:
    # search for a player whose next request time is in the past (can be explored again)
    unexplored_sum = sum_db.expired_summoner()

    # check if we found an unexplored summoner
    if unexplored_sum is None:
        print(f"\n\n{sum_db} Ran out of players, getting new ones!\n")

        # get new players from challenger leagues
        puuids = fetch_players_from_league(sum_db.continent, lolwatcher)  # this will absolutely demolish our rate limit and take ages

        # calculate initial request times
        request_times = [ReqTimeCalc.initial() for _ in puuids]
        entries = [(puuid, request_time, wait_time) for puuid, (request_time, wait_time) in zip(puuids, request_times)]

        # insert them into the database
        _, new = sum_db.put_multi(entries)

        # check if we already fetched all challengers :(
        if new == 0:
            raise Exception(f"\n\n\n\n[ERROR] {sum_db} Completely RAN OUT OF SUMMONERS\n\n")
        else:
            print("\nAdded", new, "unexplored players!")
            return explore_player(match_db, sum_db, stop_q, lolwatcher)

    else:
        unexplored_sum_id, _, wait_time = unexplored_sum

    # get the match history of the summoner
    matches = lolwatcher.match.matchlist_by_puuid(match_db.continent, unexplored_sum_id, count=100, type="ranked", queue=420)

    # get the rank of the summoner
    leagues = lolwatcher.league.by_puuid(matches[0].split('_')[0], unexplored_sum_id)
    league = next((league for league in leagues if league['queueType'] == "RANKED_SOLO_5x5"), None)

    # insert into the match database if the summoner is ranked
    if league is not None:
        total, new_inserted = match_db.put_multi([(m_id, False, league['tier'], league['rank']) for m_id in matches])
    else:
        print("unranked summoner, skipping...")
        total, new_inserted = 1, 0

    # update summoner
    next_time, wait_time = ReqTimeCalc(wait_time).step(new_inserted / total)
    sum_db.put(unexplored_sum_id, next_time, wait_time, overwrite=True)

    # if we did not insert any new matches, recurse
    if new_inserted == 0:
        return explore_player(match_db, sum_db, stop_q, lolwatcher)

    return None


# collect the players from the grandmaster leagues of all regions in our continent
def fetch_players_from_league(continent: str, lolwatcher: rw.LolWatcher) -> list[str]:
    # find regions in our continent
    puuids = []
    for region in lib.CONTINENTS_REGIONS_MAP[continent]:
        print("Getting Players of Challenger League for", region)

        # request challenger league
        chals = lolwatcher.league.challenger_by_queue(region, "RANKED_SOLO_5x5")['entries']

        # extract the summoners puuid from the entries
        [puuids.append(chal['puuid']) for chal in chals]

    return puuids
