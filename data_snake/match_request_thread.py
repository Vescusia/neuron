from pathlib import Path
from time import time
from queue import Queue
from pprint import pprint

import riotwatcher as rw

import lib
from .continent_db import MatchDB, SummonerDB
from .compressed_json_ball import CompressedJSONBall
from .league_arrow import ContinentDataset
from .reqtimecalc import ReqTimeCalc


def crawl_continent(stop_q: Queue[None], state_q: Queue[int], match_db: MatchDB, sum_db: SummonerDB, matches_path: Path, dataset: ContinentDataset, lolwatcher: rw.LolWatcher) -> None:
    # variable for incremental explored matches,
    # with incremental meaning between updates to state_q
    inc_explored_matches = 0

    # create matches_dir for JSON files
    matches_path = matches_path / match_db.continent / f"{int(time())}.xz"
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    # open lzma compressed JSON ball
    matches_ball = CompressedJSONBall(matches_path, split_every=36_000)

    while True:
        # break if we get the signal to stop
        if not stop_q.empty():
            stop_q.get()
            break

        # search for unexplored match
        while True:
            unexplored_match = match_db.unexplored_match()
            if unexplored_match is not None:
                new_match_id, ranked_score = unexplored_match
                break
            else:
                # request match history from an unexplored player
                explore_player(match_db, sum_db, stop_q, lolwatcher)
                # update explored matches
                state_q.put(inc_explored_matches)
                inc_explored_matches = 0

        # request Match from RiotAPI
        new_match = lolwatcher.match.by_id(match_db.continent, new_match_id)
        inc_explored_matches += 1

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

    # close the matches ball
    matches_ball.close()


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
    league = lolwatcher.league.by_puuid(matches[0].split('_')[0], unexplored_sum_id)[0]

    # insert into the match database
    total, new_inserted = match_db.put_multi([(m_id, False, league['tier'], league['rank']) for m_id in matches])

    # update summoner
    next_time, wait_time = ReqTimeCalc(wait_time).step(new_inserted / total)
    sum_db.put(unexplored_sum_id, next_time, wait_time)

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
