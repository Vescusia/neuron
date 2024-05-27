use clap::Parser;

use std::path::PathBuf;


#[derive(Debug, Parser)]
#[command(version, about, long_about = None, author)]
pub struct Args {
    /// The Path to the Comp Database
    #[arg(short, long, default_value_os_t = PathBuf::from("comp.redb".to_owned()))]
    pub comp_db: PathBuf,

    /// The Comp to lookup
    #[arg(long, num_args = 10, value_delimiter = ' ')]
    pub champion: Option<Vec<riven::consts::Champion>>,

    /// The Path to the Player Database
    ///
    /// If you want to get some stats about the PlayerDB, specify its path
    #[arg(short, long, default_value = None)]
    pub player_db: Option<PathBuf>,

    /// The Path to the Match Database
    ///
    /// If you want to get some stats about the MatchDB, specify its path
    #[arg(short, long, default_value = None)]
    pub match_db: Option<PathBuf>,
}