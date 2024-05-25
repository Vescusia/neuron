use clap::Parser;
use redb::{ReadableTable, ReadableTableMetadata};

mod utils;
mod cli;
mod region_db;
use region_db::RegionDB;
mod encoded_match_id;
use encoded_match_id::EncodedMatchId;
mod encoded_puuid;
include!(concat!(env!("OUT_DIR"), "/champ_to_u8.rs"));


//noinspection ALL
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args = cli::Args::parse();
    println!("Using API Key '{}'.", args.api_key);
    println!("Querying Regions {:?}.", args.region);
    println!("Starting Handles Provided: {:?}", args.starting_handle);
    
    // create and check riot api
    let riot_api = riven::RiotApi::new(args.api_key);
    riot_api.lol_status_v4()
        .get_platform_data(riven::consts::PlatformRoute::EUW1)
        .await?;

    // create puuid en- and decoder
    let mut puuid_encoder = encoded_puuid::PuuidEncoder::new();
    let mut puuid_decoder = encoded_puuid::PuuidDecoder::new();

    // create/open databases
    let mut player_db: RegionDB<&[u8], u8> = RegionDB::new("player.redb", &args.region)?;
    let mut matches_db: RegionDB<u64, bool> = RegionDB::new("match.redb", &args.region)?;

    // add starting handles
    utils::add_starting_handles(&matches_db, &args.starting_handle)?;
    
    // check if we have some matches in database
    let wtxn = matches_db.write()?;
    if matches_db.tables()
        .filter_map(|&db| wtxn.open_table(db).ok())
        .filter_map(|db| db.is_empty().ok())
        .all(|b| b)
    {
        println!("No Matches in DataBase! Please add some using the '--starting-handle' argument (--help)!");
        std::process::exit(0);
    }
    wtxn.commit()?;
    
    // create signal (ctrl-c) handler
    let stop = std::sync::Arc::new(tokio::sync::Mutex::new(false));
    let stop_t = stop.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.expect("Signal Errored (?) please report!");
        *stop_t.lock().await = true;
        println!("Preparing to stop... (this might take a minute)");
    });
    
    // variables to keep statistics on game validity
    let mut games_found: usize = 0;
    let mut games_ignored: usize = 0;
    
    let mut all_packed = std::collections::HashMap::new();
    
    
    // main loop
    'main: loop {
        // handle every region
        for &region in &args.region {
            // stop if signal received
            if *stop.lock().await {
                break 'main;
            }
            
            // get first match that has not been explored yet
            let rtxn = matches_db.read()?;
            let db = rtxn.open_table(matches_db[region])?;
            let unexplored_match = db.iter()?
                .filter_map(|i| i.ok())
                .find(|(_id, explored)| !explored.value());
            
            // if we actually found a match
            if let Some((unexplored_match_id, _)) = unexplored_match {
                println!("[{region}] Exploring new match {}", unexplored_match_id.value());
                
                // explore match
                let unexplored_match = riot_api.match_v5()
                    .get_match(region, &EncodedMatchId(unexplored_match_id.value()).to_string())
                    .await?;
                
                if let Some(u_match) = unexplored_match {
                    // check match explored
                    let wtxn = matches_db.write()?;
                    wtxn.open_table(matches_db[region])?.insert(
                        unexplored_match_id.value(),
                        true
                    )?;
                    wtxn.commit()?;
                    
                    // ignore match if it is not the classic game mode
                    if u_match.info.game_mode != riven::consts::GameMode::CLASSIC {
                        println!("\t-> Ignoring because of GameMode");
                        games_ignored += 1;
                        continue
                    }
                    games_found += 1;

                    // put players into database
                    let wtxn = player_db.write()?;
                    {
                        let mut db = wtxn.open_table(player_db[region])?;
                        for player_id in u_match.metadata.participants {
                            let puuid = puuid_decoder.decode(player_id)?;
                            
                            let player_explored = db.insert(
                               puuid,
                               0
                            )?.map(|e| e.value().max(1) - 1);
                            
                            // if the player already was in the database, 
                            // decrement his exploration by one (at least 0 though)
                            if let Some(explored) = player_explored {
                                db.insert(puuid, explored)?;
                            }
                        }
                    }
                    wtxn.commit()?;
                    
                    // TODO: put match result into database
                    // Pack the player champion picks into a 2d array
                    let mut teams = [[0; 5], [0; 5]];
                    for (i, player) in u_match.info.participants.into_iter().enumerate() {
                        // index into teams
                        let idx = match player.team_id {
                            riven::consts::Team::BLUE => 0,
                            riven::consts::Team::RED => 1,
                            t => anyhow::bail!("Invalid TeamId {t:?}!")
                        };
                        // this is a bit cursed
                        teams[idx][i % 5] = champ_to_u8(player.champion()?);
                    }
                    
                    // make sure that the teams and
                    // the champs of the teams are deterministically ordered
                    teams[0].sort_unstable(); teams[1].sort_unstable();
                    teams.sort_unstable_by(|x, y| {
                        x.iter().map(|&c| c as usize).sum::<usize>().cmp(
                            &y.iter().map(|&c| c as usize).sum()
                        )
                    });
                    // pack into u64
                    let packed = utils::pack_champs_u64(teams[0], teams[1]);
                    
                    // insert into database
                    if let Some(old_packed) = all_packed.insert(teams, packed) {
                        if old_packed != packed {
                            println!("\n\n!!! Disaster {old_packed:?}, {packed} !!!\n")
                        }
                        else {
                            println!("\n\nCollission avoided!\n")
                        }
                    }
                }
                else {
                    println!("Invalid MatchId in Database!");
                    continue;
                }
            }
            // if there are no matches left
            else {
                // explore a new player
                let rtxn = player_db.read()?;
                let db = rtxn.open_table(player_db[region])?;
                let player = db.iter()?
                    .filter_map(|x| x.ok())
                    .find(|(_id, explored)| explored.value() == 0);
                
                if let Some((player_id, _)) = player {
                    println!("[{region}] Exploring player {}", puuid_encoder.encode(player_id.value())?);
                    
                    // explore match history of player
                    let matches = riot_api.match_v5()
                        .get_match_ids_by_puuid(
                            region,
                            puuid_encoder.encode(player_id.value())?,
                            Some(100),
                            None, None, None, None, None
                        ).await?;
                    println!("\t-> Match history of {} matches", matches.len());
                    
                    // set player match count to length of matches
                    let wtxn = player_db.write()?;
                    wtxn.open_table(player_db[region])?
                        .insert(player_id.value(), (matches.len()+1) as u8)?;
                    wtxn.commit()?;

                    // add matches into database
                    let wtxn = matches_db.write()?;
                    {
                        let mut db = wtxn.open_table(matches_db[region])?;
                        for match_id in matches {
                            let id = EncodedMatchId::from(match_id).0;
                            if db.get(id)?.is_none() {
                                db.insert(id, false)?;
                            }
                        }
                    }
                    wtxn.commit()?;
                }
                else {
                    println!("Ran out of matches and players for {region}!");
                }
            }
        }
    }
    
    // finish
    println!("Compacting Databases...");
    println!("PlayerDB compacted: {}", player_db.db.compact()?);
    // println!("MatchDB compacted: {}", matches_db.db.compact()?);  // panics

    println!("Valid Games Analyzed: {games_found} <-> Invalid Games Ignored: {games_ignored} ({}% valid)", (games_found * 100) / (games_found + games_ignored));
    Ok(())
}
