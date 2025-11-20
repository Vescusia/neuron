"""
Encodes a champion ID into a single-byte integer

This encoding promises to be consistent across all patches; i.e., new champions will be assigned a new ID,
and old champions will retain theirs.
"""

from numpy import uint8

from lib import CHAMPIONS


# map champ.id to u8
# this is done by hand to keep the promise of consistency at all times
# print('\n'.join([f"    {champ['key']}: {i+1},  # {champ['id']}" for i, (_, champ) in enumerate(CHAMPIONS.items())]))
_id_to_int_map: dict[int, int] = {
    -1: 0,  # No Pick/Ban
    266: 1,  # Aatrox
    103: 2,  # Ahri
    84: 3,  # Akali
    166: 4,  # Akshan
    12: 5,  # Alistar
    799: 6,  # Ambessa
    32: 7,  # Amumu
    34: 8,  # Anivia
    1: 9,  # Annie
    523: 10,  # Aphelios
    22: 11,  # Ashe
    136: 12,  # AurelionSol
    893: 13,  # Aurora
    268: 14,  # Azir
    432: 15,  # Bard
    200: 16,  # Belveth
    53: 17,  # Blitzcrank
    63: 18,  # Brand
    201: 19,  # Braum
    233: 20,  # Briar
    51: 21,  # Caitlyn
    164: 22,  # Camille
    69: 23,  # Cassiopeia
    31: 24,  # Chogath
    42: 25,  # Corki
    122: 26,  # Darius
    131: 27,  # Diana
    119: 28,  # Draven
    36: 29,  # DrMundo
    245: 30,  # Ekko
    60: 31,  # Elise
    28: 32,  # Evelynn
    81: 33,  # Ezreal
    9: 34,  # Fiddlesticks
    114: 35,  # Fiora
    105: 36,  # Fizz
    3: 37,  # Galio
    41: 38,  # Gangplank
    86: 39,  # Garen
    150: 40,  # Gnar
    79: 41,  # Gragas
    104: 42,  # Graves
    887: 43,  # Gwen
    120: 44,  # Hecarim
    74: 45,  # Heimerdinger
    910: 46,  # Hwei
    420: 47,  # Illaoi
    39: 48,  # Irelia
    427: 49,  # Ivern
    40: 50,  # Janna
    59: 51,  # JarvanIV
    24: 52,  # Jax
    126: 53,  # Jayce
    202: 54,  # Jhin
    222: 55,  # Jinx
    145: 56,  # Kaisa
    429: 57,  # Kalista
    43: 58,  # Karma
    30: 59,  # Karthus
    38: 60,  # Kassadin
    55: 61,  # Katarina
    10: 62,  # Kayle
    141: 63,  # Kayn
    85: 64,  # Kennen
    121: 65,  # Khazix
    203: 66,  # Kindred
    240: 67,  # Kled
    96: 68,  # KogMaw
    897: 69,  # KSante
    7: 70,  # Leblanc
    64: 71,  # LeeSin
    89: 72,  # Leona
    876: 73,  # Lillia
    127: 74,  # Lissandra
    236: 75,  # Lucian
    117: 76,  # Lulu
    99: 77,  # Lux
    54: 78,  # Malphite
    90: 79,  # Malzahar
    57: 80,  # Maokai
    11: 81,  # MasterYi
    800: 82,  # Mel
    902: 83,  # Milio
    21: 84,  # MissFortune
    62: 85,  # MonkeyKing
    82: 86,  # Mordekaiser
    25: 87,  # Morgana
    950: 88,  # Naafiri
    267: 89,  # Nami
    75: 90,  # Nasus
    111: 91,  # Nautilus
    518: 92,  # Neeko
    76: 93,  # Nidalee
    895: 94,  # Nilah
    56: 95,  # Nocturne
    20: 96,  # Nunu
    2: 97,  # Olaf
    61: 98,  # Orianna
    516: 99,  # Ornn
    80: 100,  # Pantheon
    78: 101,  # Poppy
    555: 102,  # Pyke
    246: 103,  # Qiyana
    133: 104,  # Quinn
    497: 105,  # Rakan
    33: 106,  # Rammus
    421: 107,  # RekSai
    526: 108,  # Rell
    888: 109,  # Renata
    58: 110,  # Renekton
    107: 111,  # Rengar
    92: 112,  # Riven
    68: 113,  # Rumble
    13: 114,  # Ryze
    360: 115,  # Samira
    113: 116,  # Sejuani
    235: 117,  # Senna
    147: 118,  # Seraphine
    875: 119,  # Sett
    35: 120,  # Shaco
    98: 121,  # Shen
    102: 122,  # Shyvana
    27: 123,  # Singed
    14: 124,  # Sion
    15: 125,  # Sivir
    72: 126,  # Skarner
    901: 127,  # Smolder
    37: 128,  # Sona
    16: 129,  # Soraka
    50: 130,  # Swain
    517: 131,  # Sylas
    134: 132,  # Syndra
    223: 133,  # TahmKench
    163: 134,  # Taliyah
    91: 135,  # Talon
    44: 136,  # Taric
    17: 137,  # Teemo
    412: 138,  # Thresh
    18: 139,  # Tristana
    48: 140,  # Trundle
    23: 141,  # Tryndamere
    4: 142,  # TwistedFate
    29: 143,  # Twitch
    77: 144,  # Udyr
    6: 145,  # Urgot
    110: 146,  # Varus
    67: 147,  # Vayne
    45: 148,  # Veigar
    161: 149,  # Velkoz
    711: 150,  # Vex
    254: 151,  # Vi
    234: 152,  # Viego
    112: 153,  # Viktor
    8: 154,  # Vladimir
    106: 155,  # Volibear
    19: 156,  # Warwick
    498: 157,  # Xayah
    101: 158,  # Xerath
    5: 159,  # XinZhao
    157: 160,  # Yasuo
    777: 161,  # Yone
    83: 162,  # Yorick
    804: 163,  # Yunara
    350: 164,  # Yuumi
    154: 165,  # Zac
    238: 166,  # Zed
    221: 167,  # Zeri
    115: 168,  # Ziggs
    26: 169,  # Zilean
    142: 170,  # Zoe
    143: 171,  # Zyra
    904: 172,  # Zaahen
}

# change the datatype to numpy
_id_to_int_map: dict[int, uint8] = {champ: uint8(i) for champ, i in _id_to_int_map.items()}

# reverse the map
_int_to_id_map: dict[uint8, int] = {v: k for k, v in _id_to_int_map.items()}

# map champion ids to names
_id_to_name_map: dict[int, str] = {int(data['key']): name for name, data in CHAMPIONS.items()}


def name_to_int(champ_name: str) -> uint8:
    """
    :param champ_name: Name (in lib.CHAMPIONS 'id') of the champion. E.g., "Belveth", not "Bel'Veth"
    :return: encoded Champion as uint8
    """
    return _id_to_int_map[int(CHAMPIONS[champ_name]['key'])]


def id_to_int(champ_id: int) -> uint8:
    """
    :return: encoded Champion as uint8
    """
    return _id_to_int_map[champ_id]


def int_to_name(champ_int: uint8) -> str:
    """
    :return: Name of the encoded Champion
    """
    return _id_to_name_map[_int_to_id_map[champ_int]]
