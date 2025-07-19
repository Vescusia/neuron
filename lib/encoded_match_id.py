"""
Encodes a match ID into an 8-byte integer.
This encoding promises to be consistent across region additions.

The encoding is as follows:

|64..58|57..0|

|region| id |
"""

from lib import REGIONS

# define the composition of the 64-bit integer
_id_bits = 58
_region_bits = 6

# map regions to uint8
_region_int_map = {region: i for i, region in enumerate(REGIONS)}
# reverse map
_int_region_map: dict[int, str] = {v: k for k, v in _region_int_map.items()}


def to_bytes(match_id: str) -> bytes:
    region, match_id = match_id.split("_")

    # this will result in a 64-bit integer with |64..58|57..0|
    #                                           |region| id  |
    left = _region_int_map[region] << _id_bits
    encoded_match = left | int(match_id)

    # return int as big endian bytes
    return encoded_match.to_bytes(8, "big")


def to_match_id(encoded_id: bytes) -> str:
    # make int from bytes
    encoded_match = int.from_bytes(encoded_id, "big")

    # mask out and map region
    region = encoded_match >> _id_bits
    region = _int_region_map[region]
    # mask out id
    match_id = encoded_match & ((1 << _id_bits) - 1)

    return region + '_' + str(match_id)
