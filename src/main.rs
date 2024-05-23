use clap::Parser;
use heed::{byteorder, types};

mod utils;
use utils::{match_id_to_u64, u64_to_match_id};
mod cli;
mod region_db;
use region_db::RegionDB;


//noinspection ALL
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args = cli::Args::parse();
    println!("Using API Key '{}'.", args.api_key);
    println!("Querying Regions {:?}.", args.regions);
    println!("Starting Handles Provided: {:?}", args.starting_handle);

    // create and check riot api
    let riot_api = riven::RiotApi::new(args.api_key);
    riot_api.lol_status_v4()
        .get_platform_data(riven::consts::PlatformRoute::EUW1)
        .await?;

    // create puuid en- and decoder
    let mut puuid_encoder = utils::PuuidEncoder::new();
    let mut puuid_decoder = utils::PuuidDecoder::new();

    // create/open databases
    let player_dbs: RegionDB<types::Bytes, types::U8> = RegionDB::new("player_db", &args.regions)?;
    let matches_dbs: RegionDB<types::U64<byteorder::NativeEndian>, types::U8> = RegionDB::new("match_db", &args.regions)?;

    // add starting handles
    utils::add_starting_handles(&matches_dbs, &args.starting_handle)?;
    
    // check if we have some matches in database
    let rtxn = matches_dbs.rtxn()?;
    if matches_dbs.dbs()
        .filter_map(|db| db.is_empty(&rtxn).ok())
        .all(|b| b)
    {
        println!("No matches in DataBase! Please add some using the '--starting-handles' argument (--help)!");
        std::process::exit(0);
    }
    rtxn.commit()?;
    
    /*let rtxn = matches_dbs.rtxn()?;
    matches_dbs.get(&riven::consts::RegionalRoute::EUROPE).unwrap()
        .iter(&rtxn)?
        .filter_map(|x| x.ok())
        .for_each(|(id, val)| println!("Match: {}, {val}", utils::u64_to_match_id(id)));
    rtxn.commit()?;
    println!("\n");
    
    let rtxn = player_dbs.rtxn()?;
    player_dbs.get(&riven::consts::RegionalRoute::EUROPE).unwrap()
        .iter(&rtxn)?
        .filter_map(|x| x.ok())
        .for_each(|(id, val)| println!("Player: {:?}, {val}", puuid_encoder.encode(id)));
    rtxn.commit()?;
    std::process::exit(0);*/

    // main loop
    loop {
        // TODO: Signal Handling

        for region in &args.regions {
            // get first match that has not been explored yet
            let rtxn = matches_dbs.rtxn()?;
            let unexplored_match = matches_dbs.get(region).unwrap().iter(&rtxn)?
                .filter_map(|x| x.ok())
                .find(|(_id, explored)| *explored == 0);

            if let Some((unexplored_match, _)) = unexplored_match {
                println!("Exploring new match {unexplored_match:?}");
                
                // explore match
                let unexplored_match = riot_api.match_v5()
                    .get_match(*region, &u64_to_match_id(unexplored_match))
                    .await?;
                rtxn.commit()?;
                
                if let Some(u_match) = unexplored_match {
                    // check match explored
                    let mut wtxn = matches_dbs.wtxn()?;
                    matches_dbs.get(region).unwrap()
                        .put(
                            &mut wtxn, 
                            &match_id_to_u64(u_match.metadata.match_id), 
                            &1
                        )?;
                    wtxn.commit()?;
                    
                    // ignore match if it is not the classic game mode
                    if u_match.info.game_mode != riven::consts::GameMode::CLASSIC {
                        continue
                    }

                    // put players into database
                    let mut wtxn = player_dbs.wtxn()?;
                    let db = player_dbs.get(region).unwrap();
                    for player in u_match.metadata.participants {
                        let puuid = puuid_decoder.decode(player)?;
                        
                        let player_explored = db.get_or_put(
                           &mut wtxn,
                           puuid,
                           &0
                        )?;
                        
                        // if the player already was in the database, 
                        // decrement his exploration by one (at least 0 though)
                        if let Some(explored) = player_explored {
                            db.put(&mut wtxn, puuid, &(explored.max(1) - 1))?
                        }
                    }
                    wtxn.commit()?;
                    
                    // TODO: put match result into database
                }
                else {
                    println!("Invalid MatchId in Database!");
                    continue;
                }
            }
            // if there are no matches left
            else {
                rtxn.commit()?;
                
                // explore a new player
                let rtxn = player_dbs.rtxn()?;
                let player = player_dbs.get(region).unwrap().iter(&rtxn)?
                    .filter_map(|x| x.ok())
                    .find(|(_id, explored)| *explored == 0);
                
                if let Some((player_id, _)) = player {
                    println!("Exploring player {player_id:?}");
                    
                    // explore match history of player
                    let matches = riot_api.match_v5()
                        .get_match_ids_by_puuid(
                            *region,
                            puuid_encoder.encode(player_id)?,
                            Some(100),
                            None, None, None, None, None
                        ).await?;
                    println!("Match history: {matches:?}");
                    
                    // set player match count to length of matches
                    let mut wtxn = player_dbs.wtxn()?;
                    player_dbs.get(region).unwrap()
                        .put(&mut wtxn, player_id, &((matches.len()+1) as u8))?;
                    rtxn.commit()?;
                    wtxn.commit()?;

                    // add matches into database
                    let mut wtxn = matches_dbs.wtxn()?;
                    let db = matches_dbs.get(region).unwrap();
                    for match_id in matches {
                        db.get_or_put(&mut wtxn, &match_id_to_u64(match_id), &0)?;
                    }
                    wtxn.commit()?;
                }
                else {
                    println!("Ran out of matches and players for {region}!");
                    rtxn.commit()?;
                }
            }
        }
    }
}
