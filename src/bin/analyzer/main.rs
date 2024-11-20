use clap::Parser;
use redb::{ReadableTable, ReadableTableMetadata, TableHandle};
use bytesize::ByteSize;
use neuron::u8_to_champion;

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

    // check if synergy DB Path exists
    if !args.syn_db.exists() {
        eprintln!("Synergy Database {:?} does not exist!", args.syn_db);
        eprintln!("Please specify a valid Path!");
        std::process::exit(-1)
    }
    // check if matchup DB Path exists
    if !args.matchup_db.exists() {
        eprintln!("Matchup Database {:?} does not exist!", args.matchup_db);
        eprintln!("Please specify a valid Path!");
        std::process::exit(-1)
    }
    

    // analyze syn DB
    let mut self_synergies = std::collections::HashMap::<riven::consts::Champion, u32>::new();
    let db = redb::Database::open(&args.syn_db)?;
    let rtxn = db.begin_read()?;
    let start = std::time::Instant::now();
    let mut highest_total = (0., (0., 0, None));
    for table in rtxn.list_tables()? {
        // specify table
        let name = table.name();
        let table_def = redb::TableDefinition::<neuron::SynDbK, neuron::SynDbV>::new(name);
        // open table
        let table = rtxn.open_table(table_def)?;

        // sum up all self-synergies
        for (key, value) in table.iter()?.filter_map(|e| e.ok()) {
            let synergy = key.value();
            let (wins, total, _damage_diff) = value.value();

            let winrate = wins as f32 / total as f32;
            if winrate * (total as f32).sqrt() > highest_total.0 {
                highest_total.0 = winrate * (total as f32).sqrt();
                highest_total.1 = (winrate, total, Some(synergy.map(u8_to_champion).map(riven::consts::Champion::identifier)));
            }

            if synergy[0] == synergy[1] {
                let old_total = self_synergies.get(&u8_to_champion(synergy[0]));
                self_synergies.insert(u8_to_champion(synergy[0]), old_total.unwrap_or(&0) + total as u32);
            }
        }
    }
    println!("Total self-synergies: {} in {:.2}s", self_synergies.values().sum::<u32>(), start.elapsed().as_secs_f64());
    println!("Highest total: {:?}", highest_total);

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
