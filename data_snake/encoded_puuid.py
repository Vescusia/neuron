import base64


class EncodedPUUID:
    padding = "=" * (54 % 4)

    def __init__(self, puuid: str | bytes):
        if type(puuid) is bytes:
            self.puuid = puuid
        elif type(puuid) is str:
            self.puuid = base64.urlsafe_b64decode(puuid + self.padding)

    def __bytes__(self):
        return self.to_bytes()

    def to_bytes(self) -> bytes:
        return self.puuid

    def decode(self) -> str:
        return base64.b64encode(self.puuid, altchars=b"-_")[:-len(self.padding)].decode("utf-8")

    def __str__(self):
        return self.decode()
