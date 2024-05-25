use redb::ReadableTable;
use crate::encoded_match_id::EncodedMatchId;
use crate::region_db::RegionDB;


pub fn add_starting_handles(match_dbs: &RegionDB<u64, bool>, match_ids: &Vec<String>) -> anyhow::Result<()> {
    let wtxn = match_dbs.write()?;
    
    for match_id in match_ids {
        // encode id
        let id = EncodedMatchId::from(match_id);
        
        // insert match if not exists
        let mut table = wtxn.open_table(match_dbs[id.platform().to_regional()])?;
        if table.get(id.0)?.is_none() {
            table.insert(id.0, false)?;
        }
        println!("Added {match_id} as Starting Handle.")
    }
    
    wtxn.commit()?;
    Ok(())
}


pub fn pack_champs_u64(mut team1: [u8; 5], mut team2: [u8; 5]) -> u64 {
    // sort both teams to remove importance of draft order
    team1.sort_unstable(); team2.sort_unstable();
    
    // work in the champs
    let mut packed = 0;
    for champ in team1.into_iter().chain(team2.into_iter()) {
        packed ^= champ as u64;
        packed = packed.rotate_left(7);
    }
    
    packed
}
