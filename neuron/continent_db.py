from cassiopeia.data import Continent
from lmdb import Environment


class ContinentDB:
    def __init__(self, env: Environment, continent: Continent, **kwargs):
        self.continent = continent
        self.env = env
        self.db = env.open_db(str(continent).encode(), **kwargs)

    def begin(self, write: bool = False, **kwargs):
        return self.env.begin(write=write, db=self.db, **kwargs)

    def __str__(self):
        return f"RegionDB({self.continent})"
