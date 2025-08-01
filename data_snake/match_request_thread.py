from pathlib import Path
from time import time, sleep
from queue import Queue
import traceback

from numpy import uint8
import requests
import riotwatcher as rw

import lib
from lib.league_of_parquet import ContinentDatasetWriter
from .continent_db import MatchDB, SummonerDB
from .compressed_json_ball import CompressedJSONBall
from .reqtimecalc import ReqTimeCalc


def crawl_continent(stop_q: Queue[None], state_q: Queue, match_db: MatchDB, sum_db: SummonerDB, matches_path: Path, dataset: ContinentDatasetWriter, lolwatcher: rw.LolWatcher) -> None:
    # define the continent we are crawling by copying the one from the continent-specific match database
    continent = match_db.continent

    # print database state
    print(f"{sum_db.continent}: {sum_db.count()} Summoners, {match_db.count()} Matches")

    # initialize the list of unexplored matches to an empty list; will be redefined every loop
    new_match_ids = []

    # create matches_dir for JSON files
    matches_path = matches_path / continent / f"{int(time())}.gzip"
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    # open lzma compressed JSON ball
    matches_ball = CompressedJSONBall(matches_path, split_every=36_000)

    while True:
        try:
            # break if we get the signal to stop
            if not stop_q.empty():
                state_q.put((len(new_match_ids), 0.))
                break

            # fetch match history from a player
            new_match_ids, ranked_score, satisfaction = fetch_player(match_db, sum_db, lolwatcher)
            # update state thread
            state_q.put((len(new_match_ids), satisfaction))

            # explore every match
            for match_id in new_match_ids:
                # request Match from RiotAPI
                try:
                    new_match = lolwatcher.match.by_id(continent, match_id)
                except rw.ApiError as e:
                    # check if this match id is invalid
                    if e.response.status_code == 404:
                        print(f"\n[ERROR {continent}] Match {match_id} is invalid, skipping...\n")
                        # mark as explored
                        match_db.append(match_id, ranked_score)
                        continue
                    # check if the API key has become unauthorized
                    elif e.response.status_code == 401:
                        print(f"\n[ERROR {continent}] Unauthorized API Key, exiting...\n")
                        break
                    # bubble up the error if it's not recognized here
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

                # save match in the match database
                match_db.append(match_id, ranked_score)

        # catch the errors that just sometimes happen with web traffic.
        except (requests.exceptions.HTTPError, requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
            print(f"\n[ERROR {continent}]:")
            traceback.print_exception(e)
            print("\nContinuing...")
            sleep(5)
            continue

    # close the matches ball
    matches_ball.close()

    # write matches between intervals to dataset
    dataset.write_match_list()


def fetch_player(match_db: MatchDB, sum_db: SummonerDB, lolwatcher: rw.LolWatcher) -> tuple[list[str], uint8, float]:
    """
    Fetch the match history of a player.
    Fetching the match history of a player will update the summoner's request time,
    which forbids fetching their match history again for a while (such that they can play matches).

    :return: Tuple (list of new match ids, ranked score of player, satisfaction (len(list of new match ids) / total number of matches in match history))
    """

    # search for a player whose next request time is in the past (can be fetched again)
    unfetched_sum = sum_db.expired_summoner()

    # check if we found an unfetched summoner
    if unfetched_sum is None:
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
            return fetch_player(match_db, sum_db, lolwatcher)

    else:
        # unpack unfetched summoner to puuid, request time (in the past) and the last wait time
        summoner_id, _, wait_time = unfetched_sum

    # get the match history of the summoner
    matches = lolwatcher.match.matchlist_by_puuid(match_db.continent, summoner_id, count=100, type="ranked", queue=420)

    # get the rank of the summoner
    if len(matches) > 0:
        # fetch the (multiple) leagues for the summoner
        leagues = lolwatcher.league.by_puuid(matches[0].split('_')[0], summoner_id)
        # only the ranked 5 vs. 5 solo/duo rank
        league = next((league for league in leagues if league['queueType'] == "RANKED_SOLO_5x5"), None)
        # convert to uint8 ranked score
        ranked_score = lib.encoded_rank.to_int(league['tier'], league['rank']) if league is not None else None
    else:
        ranked_score = None

    # try to find fetched matches in the database
    old_matches = [match_id for match_id, _, _ in match_db.get_multi(matches)]

    # select only new, not yet explored, matches
    new_matches = [match_id for match_id in matches if match_id not in old_matches]

    # calculate satisfaction (the percentage of matches in the history that were not yet in the match database)
    if ranked_score is not None:
        satisfaction = len(new_matches) / len(matches)
    else:
        print("unranked summoner, skipping...")
        satisfaction = 0.

    # update summoner next request time of the summoner in the database
    next_time, wait_time = ReqTimeCalc(wait_time).step(satisfaction)
    sum_db.put(summoner_id, next_time, wait_time, overwrite=True)

    # if we did not find any new matches, recurse and fetch another player
    if satisfaction == 0.:
        new_matches, ranked_score, satisfaction = fetch_player(match_db, sum_db, lolwatcher)
        return new_matches, ranked_score, satisfaction / 2  # take average of both satisfactions

    return new_matches, ranked_score, satisfaction


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
