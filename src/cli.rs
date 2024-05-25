use clap::Parser;
use riven::consts::RegionalRoute::{self, *};

#[derive(Debug, Parser)]
#[command(version, about, long_about = None, author)]
pub struct Args {
    /// Your Riot Games API Key
    #[arg()]
    pub api_key: String,
    
    /// The regions to work on
    /// (https://docs.rs/riven/latest/riven/consts/enum.RegionalRoute.html)
    #[arg(short, long, default_values_t = [AMERICAS, ASIA, EUROPE, SEA])]
    pub region: Vec<RegionalRoute>,
    
    /// MatchIds to be used as "Starting Handles", 
    /// as in providing some matches for an empty database
    /// so that the scraping process can begin.
    /// 
    /// Get Ids with https://developer.riotgames.com/apis#match-v5/GET_getMatch.
    /// 
    /// If possible, add a good variance of Ids (ranks, regions), 
    /// to diversify the scraping process.
    #[arg(long)]
    pub starting_handle: Vec<String>
}