import riotwatcher as _rw

data_dragon = _rw.LolWatcher(api_key='not-used').data_dragon

# project wide static variables
CONTINENTS = ["EUROPE", "AMERICAS", "ASIA", "SEA"]

# !!! new regions may only be appended to this list !!!
# to preserve consistency with older data instances
REGIONS = ["EUW1", "EUN1", "RU", "TR1", "NA1", "BR1", "LA1", "LA2", "OC1", "ME1", "JP1", "KR", "TW2", "VN2", "SG2", "PH2", "TH2"]

DEPRECATED_REGIONS = ["SG2", "PH2", "TH2"]

CONTINENTS_REGIONS_MAP = {"EUROPE": ["EUW1", "EUN1", "RU", "TR1", "ME1"], "AMERICAS": ["NA1", "BR1", "LA1", "LA2"], "ASIA": ["KR", "JP1"], "SEA": ["OC1", "TW2", "VN2", "SG2", "PH2", "TH2"]}

LATEST_VERSION = data_dragon.versions_all()[0]

CHAMPIONS = data_dragon.champions(version=LATEST_VERSION)['data']

TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]
DIVISIONS = ["I", "II", "III", "IV"]

import lib.encoded_patch
from .encoded_patch import ALL_PATCHES

import lib.encoded_champ_id
import lib.encoded_rank
import lib.encoded_match_id
import lib.league_of_parquet
