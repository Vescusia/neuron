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


/// Packs two teams deterministically into an u64.
/// 
/// The teams have to be ordered in a deterministic manner externally!
pub fn pack_teams_u64(mut team1: [u8; 5], mut team2: [u8; 5]) -> u64 {
    // sort both team comps to remove importance of draft order
    team1.sort_unstable(); team2.sort_unstable();
    
    // work in the champs
    let mut packed = 0;
    for champ in team1.into_iter().chain(team2.into_iter()) {
        packed ^= champ as u64;
        packed = packed.rotate_left(7);
    }
    
    packed
}

pub fn cmp_comps_u8(team1: [u8; 5], team2: [u8; 5]) -> std::cmp::Ordering {
    let t1_value: u16 = team1.map(|c| c as u16).into_iter().sum();
    let t2_value = team2.map(|c| c as u16).into_iter().sum();
    t1_value.cmp(&t2_value)
}


#[cfg(test)]
mod tests {
    use rand::prelude::SliceRandom;
    use rand::Rng;
    use riven::consts::Champion;
    use crate::champ_to_u8;
    use super::*;
    
    
    #[test]
    fn test_pack_champs() {
        // init teams
        let mut team1 = [
            champ_to_u8(Champion::URGOT),
            champ_to_u8(Champion::YASUO),
            champ_to_u8(Champion::YUUMI),
            champ_to_u8(Champion::ZERI),
            champ_to_u8(Champion::DIANA)
        ];
        let mut team2 = [
            champ_to_u8(Champion::FIORA),
            champ_to_u8(Champion::AURELION_SOL),
            champ_to_u8(Champion::TWITCH),
            champ_to_u8(Champion::BRAUM),
            champ_to_u8(Champion::LEE_SIN)
        ];
        
        // initial value
        let start = pack_teams_u64(team1, team2);
        
        // start shuffling
        let mut rng = rand::thread_rng();
        for i in 0..100_000 {
            team1.shuffle(&mut rng);
            team2.shuffle(&mut rng);
            
            if i % 2 == 0 {
                assert_eq!(start, pack_teams_u64(team1, team2))
            } else {
                assert_eq!(start, pack_teams_u64(team2, team1))
            }
        }
        
        // start random team testing
        for _ in 0..1_000_000 {
            let (rand_team1, rand_team2) = gen_rand_teams(&mut rng);
            let packed = pack_teams_u64(rand_team1, rand_team2);
            
            assert_ne!(packed, start, "Packed Collision (could be coincidence):\n{team1:?} vs. {team2:?}\n{rand_team1:?} vs. {rand_team2:?}");
        }
    }
    
    fn gen_rand_teams(rng: &mut rand::rngs::ThreadRng) -> ([u8; 5], [u8; 5]) {
        // generate
        let mut teams = [0; 10];
        for i in 0..teams.len() {
            let num = loop {
                let num: u8 = rng.gen_range(0..=168);
                if !teams.iter().any(|c| *c == num) {
                    break num
                }
            };
            teams[i] = num
        }
        // extract
        let (team1, team2) = teams.split_first_chunk::<5>().unwrap();
        let (team2, _) = team2.split_first_chunk::<5>().unwrap();
        (*team1, *team2)
    }
}
