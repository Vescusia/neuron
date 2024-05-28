use clap::Parser;
use redb::{ReadableTable, ReadableTableMetadata};
use riven::consts::{Champion, RegionalRoute};

use std::sync::Arc;


mod cli;

use neuron::{region_db::RegionDB, encoded_match_id::EncodedMatchId};
use neuron::batch_match_requester::BatchMatchRequester;


const COMP_DB_TABLE: redb::TableDefinition<neuron::CompDbK, neuron::CompDbV> = redb::TableDefinition::new("comps");


//noinspection ALL
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args = cli::Args::parse();
    println!("Using API Key '{}'.", args.api_key);
    println!("Querying Regions {:?}.", args.region);
    println!("Starting Handles Provided: {:?}", args.starting_handle);
    
    // create and check riot api
    let riot_api = std::sync::Arc::new(riven::RiotApi::new(args.api_key));
    riot_api.lol_status_v4()
        .get_platform_data(riven::consts::PlatformRoute::EUW1)
        .await?;

    // create/open databases
    let player_db = Arc::new(RegionDB::new("player.redb", &args.region)?);
    let match_db = Arc::new(RegionDB::new("match.redb", &args.region)?);
    let comp_db = Arc::new(redb::Database::create("comp.redb")?);

    // add starting handles
    neuron::add_starting_handles(&match_db, &args.starting_handle)?;
    
    // check if we have some matches in database
    let wtxn = match_db.write()?;
    if match_db.tables()
        .filter_map(|&db| wtxn.open_table(db).ok())
        .filter_map(|db| db.is_empty().ok())
        .all(|b| b)
    {
        println!("No Matches in DataBase! Please add some using the '--starting-handle' argument (--help)!");
        std::process::exit(0);
    }
    wtxn.commit()?;
    
    // create signal (ctrl-c) handler
    let stop = Arc::new(tokio::sync::Mutex::new(false));
    let stop_t = stop.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.expect("Signal Errored (?) please report!");
        *stop_t.lock().await = true;
        println!("\nPreparing to stop... (this might take a minute)");
    });
    
    // to keep track of games being received
    // spawn threads for each region
    let mut threads = Vec::with_capacity(args.region.len());
    for region in args.region {
        let api = riot_api.clone();
        let player_db = player_db.clone();
        let match_db = match_db.clone();
        let comp_db = comp_db.clone();
        let stop = stop.clone();
        threads.push((
            tokio::task::spawn(async move {
                region_scrape_thread(api, player_db, match_db, comp_db, stop, region).await
            }),
            region
        ));
    }
    
    // wait on threads
    let mut games_found = 1;
    let mut games_ignored = 1;
    for (thread, region) in threads {
        let (found, ignored) = thread.await??;
        games_found += found; games_ignored += ignored;
        println!("{region} Finished.")
    }
    
    // finish
    println!("Classic Games Scraped: {games_found} <-> Non Classic Games Ignored: {games_ignored} ({}% valid)", (games_found * 100) / (games_found + games_ignored));
    Ok(())
}



// A region specific Scrape Thread
async fn region_scrape_thread<'a>(
    api: Arc<riven::RiotApi>, 
    player_db: Arc<RegionDB<neuron::PlayerDbK, neuron::PlayerDbV>>,
    match_db: Arc<RegionDB<neuron::MatchDbK, neuron::MatchDbV>>, 
    comp_db: Arc<redb::Database>, stop: Arc<tokio::sync::Mutex<bool>>,
    region: RegionalRoute
) -> anyhow::Result<(usize, usize)> 
{
    // create puuid en- and decoder
    let mut puuid_encoder = neuron::encoded_puuid::PuuidEncoder::<78>::new();
    let mut puuid_decoder = neuron::encoded_puuid::PuuidDecoder::<58>::new();
    let mut batch_requester = BatchMatchRequester::new(33, api.clone(), region);
    
    // Keep track of scrape statistics
    let mut games_found = 0;
    let mut games_ignored = 0;
    
    // main loop
    loop {
        // stop if signal received
        if *stop.lock().await {
            break Ok((games_found, games_ignored))
        }
        
        // get matches that have not been explored yet
        let rtxn = match_db.read()?;
        let db = rtxn.open_table(match_db[region])?;
        let batch = batch_requester
            .request(
                db.iter()?
                    .filter_map(|m| m.ok())
                    .filter(|(_, explored)| !explored.value())
                    .map(|(u_match, _)| u_match.value())
                    .map(EncodedMatchId)
            ).await;
        
        // if we actually found some matches
        if let Some(unexplored_matches) = batch {
            // iterate over the matches
            for (unexplored_match, unexplored_match_id) in unexplored_matches {
                // check for error
                let unexplored_match = match unexplored_match {
                    Ok(m) => m,
                    Err(e) => anyhow::bail!("{e}")
                };
                let unexplored_match_id = *unexplored_match_id;
                
                if let Some(u_match) = unexplored_match {
                    // check match explored
                    let wtxn = match_db.write()?;
                    wtxn.open_table(match_db[region])?.insert(
                        unexplored_match_id,
                        true
                    )?;
                    wtxn.commit()?;
    
                    // ignore match if it is not the classic game mode
                    if u_match.info.game_mode != riven::consts::GameMode::CLASSIC {
                        games_ignored += 1;
                        continue
                    }
                    games_found += 1;
    
                    // put players into database
                    let wtxn = player_db.write()?;
                    {
                        let mut db = wtxn.open_table(player_db[region])?;
                        for player_id in &u_match.metadata.participants {
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
    
                    // Pack the player champion picks into a 2d array
                    let mut teams = [[Champion::NONE; 5], [Champion::NONE; 5]];
                    for (i, player) in u_match.info.participants.iter().enumerate() {
                        // index into teams
                        let idx = match player.team_id {
                            riven::consts::Team::BLUE => 0,
                            riven::consts::Team::RED => 1,
                            t => anyhow::bail!("Invalid TeamId {t:?}!")
                        };
                        teams[idx][i % 5] = player.champion()?;
                    }
                    // pack into u64
                    let packed = neuron::packed_comp::PackedComp::pack(teams[0], teams[1]);
                
                    // swap teams to remain deterministically only based on team comps
                    let mut result = if u_match.info.teams[0].win { 1 } else { -1 };
                    if packed.ord == std::cmp::Ordering::Less {
                        result *= -1;
                    }
    
                    // insert into database
                    let wtxn = comp_db.begin_write()?;
                    {
                        let packed = packed.packed;
                        let mut db = wtxn.open_table(COMP_DB_TABLE)?;
                        // get maybe existing
                        let old = db.get(packed)?
                            .map(|m| m.value());
                        // update or insert new
                        if let Some((balance, games)) = old {
                            db.insert(packed, (balance+result, games+1))?;
                        } 
                        else {
                            db.insert(packed, (result, 1))?;
                        }
                    }
                    wtxn.commit()?;
                }
                else {
                    println!("\nInvalid MatchId {} in Database!", unexplored_match_id);
        
                    // remove invalid match
                    let wtxn = match_db.write()?;
                    wtxn.open_table(match_db[region])?
                        .remove(unexplored_match_id)?;
                    wtxn.commit()?;
                    println!("\t-> Removed.\n");
                    
                    continue;
                }
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
                //println!("\n[{region}] Exploring player {}", puuid_encoder.encode(player_id.value())?);
                
                // explore match history of player
                let matches = api.match_v5()
                    .get_match_ids_by_puuid(
                        region,
                        puuid_encoder.encode(player_id.value())?,
                        Some(100),
                        None, None, None, None, None
                    ).await?;
                //println!("\t-> Match history of {} matches", matches.len());
                
                // set player match count to length of matches
                let wtxn = player_db.write()?;
                wtxn.open_table(player_db[region])?
                    .insert(player_id.value(), (matches.len()+1) as u8)?;
                wtxn.commit()?;
    
                // add matches into database
                let wtxn = match_db.write()?;
                {
                    let mut db = wtxn.open_table(match_db[region])?;
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
                println!("\nRan out of matches and players for {region}!");
            }
        }
    }
}
