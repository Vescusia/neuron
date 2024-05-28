use clap::Parser;
use redb::{ReadableTable, ReadableTableMetadata, TableHandle};
use bytesize::ByteSize;

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


    // check if Match DB Path exists
    if !args.comp_db.exists() {
        eprintln!("Comp Database {:?} does not exist!", args.comp_db);
        eprintln!("Please specify a valid Path!");
        std::process::exit(-1)
    }

    // print Comp DB stats
    print_db_stats(&args.comp_db)?;

    // analyze Comp DB
    let db = redb::Database::open(&args.comp_db)?;
    let rtxn = db.begin_read()?;
    for table in rtxn.list_tables()? {
        // specify table
        let name = table.name();
        let table_def = redb::TableDefinition::<neuron::CompDbK, neuron::CompDbV>::new(name);
        // open table
        let table = rtxn.open_table(table_def)?;

        // analyze table
        println!("Total greater than 1 Scan:");
        for (key, value) in table.iter()?.filter_map(|e| e.ok()) {
            let packed_comp = key.value();
            let (balance, total) = value.value();

            if total > 1 {
                println!("\t-> {packed_comp:016x} => {balance} / {total}")
            }
        }

        // get comp
        if let Some(teams) = &args.champion {
            // pack teams
            let (&team0, team1) = teams.split_first_chunk::<5>().unwrap();
            let (&team1, _) = team1.split_first_chunk::<5>().unwrap();
            let packed = neuron::packed_comp::PackedComp::pack(team0, team1);
            
            print!("\nSearching Comp: ");
            print_champ_comp(teams);
            let res = table.get(packed.packed)?;
            if let Some((mut balance, total)) = res.map(|res| res.value()) {
                if packed.ord == std::cmp::Ordering::Less {
                    balance *= -1;
                }
                println!("\t-> Balance: {balance}, Total: {total} ({} ∆%)", (balance as f32) * 100. / (total as f32))
            } else {
                println!("\t-> Not Found.")
            }
        }
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
    for table in rtxn.list_tables()? {
        let name = table.name();
        println!("\t-> Table: [{name}]");
        let table = rtxn.open_untyped_table(table)?;
        let stats = table.stats()?;

        println!("\t\t-> Number of entries:    {}", table.len()?);
        println!("\t\t-> Total Storage Bytes:  {}", ByteSize(stats.stored_bytes()));
        println!("\t\t-> Total Metadata Bytes: {}", ByteSize(stats.metadata_bytes()));
    }

    // compact DB
    if repaired {
        println!("\t-> got compacted: {:?}\n", db.compact()?);
    }

    Ok(())
}


fn print_champ_comp(champs: &[riven::consts::Champion]) {
    print!("[ ");
    for (i, champ) in champs.iter().enumerate() {
        print!("{} ", champ.name().unwrap());
        if i == 4 {
            print!("] vs. [ ")
        }
    }
    println!("]");
}
