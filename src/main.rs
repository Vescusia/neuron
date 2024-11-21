use clap::Parser;
use redb::{ReadableTable, ReadableTableMetadata};
use riven::consts::RegionalRoute;

use std::sync::Arc;


mod cli;

use neuron::{region_db::RegionDB, encoded_match_id::EncodedMatchId};
use neuron::batch_match_requester::BatchMatchRequester;
use neuron::champ_to_u8;
use neuron::participant_combinations::*;


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
    let syn_db = Arc::new(redb::Database::create("synergy.redb")?);
    let matchup_db = Arc::new(redb::Database::create("matchup.redb")?);

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
        let syn_db = syn_db.clone();
        let matchup_db = matchup_db.clone();
        let stop = stop.clone();
        threads.push((
            tokio::task::spawn(async move {
                region_scrape_thread(api, player_db, match_db, syn_db, matchup_db, stop, region).await
            }),
            region
        ));
    }

    // wait on threads
    let start = std::time::Instant::now();
    let mut games_found = 0;
    let mut games_ignored = 0;
    for (thread, region) in threads {
        let (found, ignored) = thread.await??;
        games_found += found; games_ignored += ignored;
        println!("{region} Finished.")
    }

    // finish
    println!("Classic Games Scraped: {games_found} <-> Non Classic Games Ignored: {games_ignored} ({}% valid, ({:.3}/s))", ((games_found+1) * 100) / (games_found + 1 + games_ignored), (games_found + games_ignored) as f64 / start.elapsed().as_secs_f64());
    Ok(())
}



// A region specific Scrape Thread
async fn region_scrape_thread<'a>(
    api: Arc<riven::RiotApi>,
    player_db: Arc<RegionDB<neuron::PlayerDbK, neuron::PlayerDbV>>,
    match_db: Arc<RegionDB<neuron::MatchDbK, neuron::MatchDbV>>,
    syn_db: Arc<redb::Database>,
    matchup_db: Arc<redb::Database>,
    stop: Arc<tokio::sync::Mutex<bool>>,
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
                    if u_match.info.game_mode != riven::consts::GameMode::CLASSIC || u_match.info.queue_id != riven::consts::Queue::SUMMONERS_RIFT_5V5_RANKED_SOLO {
                        games_ignored += 1;
                        continue
                    }
                    games_found += 1;

                    // update/put players into database
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

                    // create table definition for this game version
                    let patch_table = redb::TableDefinition::<neuron::SynDbK, neuron::SynDbV>::new(&u_match.info.game_version);
                    
                    // get combinations of participants
                    let participants = &u_match.info.participants;
                    let participant_combs = ParticipantCombinations(participants);
                    
                    // open matchup database
                    let wtxn = matchup_db.begin_write()?;
                    {
                        // open patch table
                        let mut db = wtxn.open_table(patch_table)?;
                        // iterate over matchups
                        for (p0, p1) in participant_combs.matchups() {
                            // get gold diff
                            let mut gold_diff = p0.gold_earned - p1.gold_earned;
                            
                            // pack champs
                            let mut packed_p = [champ_to_u8(p0.champion()?), champ_to_u8(p1.champion()?)];
                            let mut win = p0.win;
                            if packed_p[0] < packed_p[1] {
                                win = !win;
                                packed_p.swap(0, 1);
                                gold_diff *= -1;
                            }
                                
                            // get (potential) old value
                            let old = db.get(packed_p)?
                                .map(|v| v.value());

                            // insert into database
                            if let Some((wins, total, gold_diff_total)) = old {
                                db.insert(packed_p, (
                                    wins + win as u16,
                                    total+1,
                                    gold_diff_total + gold_diff,
                                ))?;
                            } else {
                                db.insert(packed_p, (
                                    win as u16, 
                                    1,
                                    gold_diff
                                ))?;
                            }
                        }
                    }
                    wtxn.commit()?;
                    
                    // open synergy database
                    let wtxn = syn_db.begin_write()?;
                    {
                        // open patch table
                        let mut db = wtxn.open_table(patch_table)?;
                        // iterate over synergies
                        for (p0, p1) in participant_combs.synergies() {
                            // get dmg diff
                            let dmg_diff = p0.total_damage_dealt_to_champions - p1.total_damage_dealt_to_champions;
                            let mut dmg_diff = dmg_diff >> 3;  // in 8-increments
                            
                            // pack champs
                            let mut packed_p = [champ_to_u8(p0.champion()?), champ_to_u8(p1.champion()?)];
                            if packed_p[0] < packed_p[1] {
                                packed_p.swap(0, 1);
                                dmg_diff *= -1;
                            }
                            
                            // get (potential) old value
                            let old = db.get(packed_p)?
                                .map(|v| v.value());

                            // insert into database
                            if let Some((wins, total, dmg_diff_total)) = old {
                                db.insert(packed_p, (
                                    wins + p0.win as u16,
                                    total+1,
                                    dmg_diff_total + dmg_diff
                                ))?;
                            } else {
                                db.insert(packed_p, (
                                    p0.win as u16, 
                                    1, 
                                    dmg_diff
                                ))?;
                            }
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
                // explore match history of player
                let matches = api.match_v5()
                    .get_match_ids_by_puuid(
                        region,
                        puuid_encoder.encode(player_id.value())?,
                        Some(100),
                        None, 
                        Some(riven::consts::Queue::SUMMONERS_RIFT_5V5_RANKED_SOLO), 
                        None, None, None
                    ).await?;
                //println!("\t-> Match history of {} matches", matches.len());

                // add matches into database
                let mut unexplored = 0;
                let wtxn = match_db.write()?;
                {
                    let mut db = wtxn.open_table(match_db[region])?;
                    for match_id in matches {
                        let id = EncodedMatchId::from(match_id).0;
                        if db.get(id)?.is_none() {
                            unexplored += 1;
                            db.insert(id, false)?;
                        }
                    }
                }
                wtxn.commit()?;

                // set player match count to length of matches
                let wtxn = player_db.write()?;
                wtxn.open_table(player_db[region])?
                    .insert(player_id.value(), (unexplored+1) as u8)?;
                wtxn.commit()?;
            }
            else {
                println!("\nRan out of matches and players for {region}!");
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        }
    }
}
