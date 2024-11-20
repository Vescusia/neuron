use redb::ReadableTable;
use crate::region_db::RegionDB;


include!(concat!(env!("OUT_DIR"), "/champ_to_u8.rs")); include!(concat!(env!("OUT_DIR"), "./u8_to_champ.rs"));

pub mod encoded_match_id;
pub mod encoded_puuid;
pub mod region_db;
pub mod packed_comp;
pub mod batch_match_requester;
pub mod participant_combinations;


// Key and Value types for Databases
pub type PlayerDbK = &'static [u8]; pub type PlayerDbV = u8;
pub type MatchDbK = u64; pub type MatchDbV = bool;
pub type CompDbK = u64; pub type CompDbV = (i16, u16);
pub type SynDbK = [u8; 2]; pub type SynDbV = (u16, u16, i32);


pub fn add_starting_handles(match_dbs: &RegionDB<u64, bool>, match_ids: &Vec<String>) -> anyhow::Result<()> {
    let wtxn = match_dbs.write()?;

    for match_id in match_ids {
        // encode id
        let id = encoded_match_id::EncodedMatchId::from(match_id);

        // insert match if not exists
        let mut table = wtxn.open_table(match_dbs[id.platform().to_regional()])?;
        if table.get(id.0)?.is_none() {
            table.insert(id.0, false)?;
        }
        println!("Added {match_id} for {} as Starting Handle.", id.platform().to_regional())
    }

    wtxn.commit()?;
    Ok(())
}
