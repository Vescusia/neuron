use clap::Parser;

use std::path::PathBuf;


#[derive(Debug, Parser)]
#[command(version, about, long_about = None, author)]
pub struct Args {
    #[clap(subcommand)]
    pub analyze: Command,

    /// The Champion to lookup
    #[arg()]
    pub champion0: riven::consts::Champion,

    #[arg()]
    /// Optional 2nd Champion to look up.
    /// Passing 1 Champion will give you all possible pairings,
    /// Passing 2 Champions will give you that specific pairing.
    pub champion1: Option<riven::consts::Champion>,

    /// The patch to analyze
    #[arg(short, long, default_value_t = Patch::all)]
    pub patch: Patch,

    /// The Path to the Player Database
    ///
    /// If you want to get some stats about the PlayerDB, specify its path
    #[arg(long, default_value = None)]
    pub player_db: Option<PathBuf>,

    /// The Path to the Match Database
    ///
    /// If you want to get some stats about the MatchDB, specify its path
    #[arg(long, default_value = None)]
    pub match_db: Option<PathBuf>,
}

#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// Analyze found Synergies
    Synergy {
        /// The Path to the Synergy Database
        #[arg(short, long, default_value_os_t = PathBuf::from("synergy.redb".to_owned()))]
        syn_db: PathBuf,
    },
    /// Analyze found Matchups
    Matchup {
        /// The Path to the Matchup Database
        #[arg(short, long, default_value_os_t = PathBuf::from("matchup.redb".to_owned()))]
        matchup_db: PathBuf,
    },
}

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum Patch{
    latest,
    all
}

impl std::fmt::Display for Patch{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}