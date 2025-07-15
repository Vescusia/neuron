import cassiopeia

from syn_db import SynergyDB, CodedSynergy

path = "../db/synergies/syn_db"
db = SynergyDB(path)

champs = cassiopeia.get_champions("EUW")
champ = champs[9]

with db.begin(cassiopeia.Patch.latest("EUW")) as txn:
    for syn in CodedSynergy.all(champ, db):
        found = txn.get(syn.to_bytes()[0])
        print(syn, found)
