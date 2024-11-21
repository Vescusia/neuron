use std::collections::HashMap;
use clap::Parser;
use redb::{ReadableTable, ReadableTableMetadata, TableHandle};
use bytesize::ByteSize;

use neuron::{champ_to_u8, u8_to_champion};

mod cli;


fn main() -> anyhow::Result<()> {
    let args = cli::Args::parse();

    // print Player DB stats
    if let Some(player) = &args.player_db {
        print_db_stats(player)?;
    }

    // print Match DB stats
    if let Some(match_db) = &args.match_db {
        print_db_stats(match_db)?;
    }

    // check if DB Paths exists and open database
    let mut db = match &args.analyze {
        cli::Command::Matchup { matchup_db } => {
            if !matchup_db.exists() {
                eprintln!("Matchup Database {:?} does not exist!", matchup_db);
                eprintln!("Please specify a valid Path!");
                std::process::exit(-1)
            } else {
                redb::Database::open(matchup_db)?
            }
        },
        cli::Command::Synergy { syn_db } => {
            if !syn_db.exists() {
                eprintln!("Synergy Database {:?} does not exist!", syn_db);
                eprintln!("Please specify a valid Path!");
                std::process::exit(-1)
            } else {
                redb::Database::open(syn_db)?
            }
        }
    };

    // repair database
    print!("Checking integrity...");
    if !db.check_integrity()? {
        println!("\rDatabase repaired.                ");
    } else {
        println!("\rDatabase integrate.               ");
    }

    // collect tables
    let rtxn = db.begin_read()?;
    let mut tables: Vec<_> = rtxn.list_tables()?.collect();
    tables.sort_by(|t0, t1| {
        match version_compare::compare(t1.name(), t0.name()).expect("Invalid Table Names!") {
            version_compare::Cmp::Gt => std::cmp::Ordering::Greater,
            version_compare::Cmp::Lt => std::cmp::Ordering::Less,
            version_compare::Cmp::Eq => std::cmp::Ordering::Equal,
            _ => { panic!("Versions have to be either gt, lt or eq!"); }
        }
    });

    // latest or all patches?
    if matches!(args.patch, cli::Patch::latest) {
        let latest = tables.first().expect("Database is empty or corrupted!").name();
        let latest = version_compare::Version::from(latest).expect("Invalid Table Name!");
        println!("{latest:?}");

        // find the first i patches which are still on the same latest "major" patch
        let mut i = 0;
        for table in tables.iter() {
            let version = version_compare::Version::from(table.name()).expect("Invalid Table Name!");
            if latest.part(0) == version.part(0) && version.part(1) == latest.part(1) {
                i += 1;
            } else {
                break
            }
        }
        tables.truncate(i);
    }
    println!("Using versions {:?}.", tables.iter().map(|t| t.name()).collect::<Vec<_>>());

    // collect data
    let start = std::time::Instant::now();
    let mut pairing_map = HashMap::new();
    for version in tables {
        // open table
        let rtxn = db.begin_read()?;
        let db = rtxn.open_table(redb::TableDefinition::<neuron::SynDbK, neuron::SynDbV>::new(version.name()))?;

        // iterate over values
        for pairing in db.iter()? {
            // unpack values
            let (pairing, (wins, total, aux)) = {
                let (pairing, value) = pairing?;
                (pairing.value(), value.value())
            };

            // make sure the pairing includes the specified champions
            if let Some(champ1) = args.champion1 {
                let champ1 = champ_to_u8(champ1);
                if champ1 != pairing[0] && champ1 != pairing[1] {
                    continue;
                }
            }
            let champ0 = champ_to_u8(args.champion0);
            if champ0 != pairing[0] && champ0 != pairing[1] {
                continue;
            }

            // insert result
            let (wins, total, aux) = (wins as u32, total as u32, aux as i64);
            let old = pairing_map.get(&pairing);
            if let Some((o_wins, o_total, o_aux)) = old {
                pairing_map.insert(pairing, (o_wins + wins, o_total + total, o_aux + aux));
            } else {
                pairing_map.insert(pairing, (wins, total, aux));
            }
        }
    }

    // calculate total games
    let total_specific_games = pairing_map.values().map(|(_, t, _)| t).sum::<u32>() / 5;

    // collect and order result
    let mut pairings = pairing_map.into_iter().collect::<Vec<_>>();
    for (pairing, (wins, total, aux)) in pairings.iter_mut() {
        if pairing[0] != champ_to_u8(args.champion0) {
            pairing.swap(0, 1);
            if matches!(&args.analyze, cli::Command::Matchup { .. }) {
                *wins = *total - *wins;
                *aux *= -1;
            }
        }
    }
    pairings.sort_by(|(_, v0), (_, v1)| {
        ((v0.0 << 8) / v0.1).cmp(&((v1.0 << 8) / v1.1))
    });
    println!("Collected all pairings in {:.2}s\n\t{total_specific_games} games", start.elapsed().as_secs_f64());

    // output result
    println!("{:12} | {:5} | {:4} | {:5}", "Champ-Pair", "Win ‰", "Amnt", "AUX");
    for (pairing, (wins, total, aux)) in pairings {
        let pairing = pairing.map(u8_to_champion).map(riven::consts::Champion::identifier).map(Option::unwrap);
        print!("{:12} | ", pairing[1]);
        println!("{:4}‰ | {:4} | {:5}", wins * 1000 / total, total, aux / (total as i64));
    }

    Ok(())
}


fn print_db_stats(path: &std::path::Path) -> anyhow::Result<()> {
    // open database
    println!("DB: [{:?}]", path);
    let mut db = redb::Database::open(path)?;
    // repair DB
    let repaired = db.check_integrity()?;
    println!("\t-> was integrate (else repaired): {}", repaired);

    // read table stats
    let rtxn = db.begin_read()?;
    if rtxn.list_tables()?.next().is_none() {
        println!("\t- no tables in {:?}", path);
    }
    for table in rtxn.list_tables()? {
        let name = table.name();
        println!("\t-> Table: [{name}]");
        let table = rtxn.open_untyped_table(table)?;
        let stats = table.stats()?;

        println!("\t\t-> Number of entries:    {}", table.len()?);
        println!("\t\t-> Total Storage Bytes:  {}", ByteSize(stats.stored_bytes()));
        println!("\t\t-> Total Metadata Bytes: {}", ByteSize(stats.metadata_bytes()));
    }
    rtxn.close()?;

    // compact DB
    if repaired {
        println!("\t-> got compacted: {:?}\n", db.compact()?);
    }

    Ok(())
}
