import cassiopeia

from continent_db import ContinentDB


class MatchIdCoder:
    id_bits = 58
    region_bits = 6

    def __init__(self, db: ContinentDB):
        assert (self.id_bits + self.region_bits) % 8 == 0

        # create region encoding
        region_encoding = {region: i+1 for i, region in enumerate(cassiopeia.Region)}

        # get actual region encoding from database or insert ours if not exists
        with db.begin(write=True) as txn:
            for region in cassiopeia.Region:
                encoded = txn.get(str(region).encode())
                if encoded:
                    region_encoding[region] = int.from_bytes(encoded, "big")
                else:
                    txn.put(str(region).encode(), region_encoding[region].to_bytes(1, "big"))

        self.region_encoding = region_encoding
        self.region_encoding_inv = dict(zip(region_encoding.values(), region_encoding.keys()))

    def encode(self, match_id: str, region: cassiopeia.Region) -> bytes:
        # this will result in a 64-bit integer with |64..58|57..0|
        #                                           |region| id  |
        left = self.region_encoding[region] << self.id_bits
        num = left | int(match_id)
        return num.to_bytes(int((self.id_bits + self.region_bits) / 8), "big")

    def decode(self, encoded_match_id: bytes) -> tuple[int, cassiopeia.Region]:
        encoded_match_id = int.from_bytes(encoded_match_id, "big")
        # mask and map region
        region = encoded_match_id >> self.id_bits
        region = self.region_encoding_inv[region]
        # mask id
        match_id = encoded_match_id & ((1 << self.id_bits) - 1)

        return match_id, region
