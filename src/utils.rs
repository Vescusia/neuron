use std::str::FromStr;
use base64::Engine;
use crate::region_db::RegionDB;


#[derive(Debug)]
pub struct PuuidDecoder {
    buf: [u8; 58]
}

impl PuuidDecoder {
    pub const fn new() -> Self {
        Self{
            buf: [0u8; 58]
        }
    }

    pub fn decode<S: AsRef<[u8]>>(&mut self, input: S) -> anyhow::Result<&[u8]> {
        let read = base64::prelude::BASE64_URL_SAFE_NO_PAD.decode_slice(input, &mut self.buf)?;
        if read != self.buf.len() {
            anyhow::bail!("Invalid decode buffer len!")
        }
        Ok(&self.buf)
    }
}

pub struct PuuidEncoder {
    buf: [u8; 78]
}

impl PuuidEncoder {
    pub const fn new() -> Self {
        Self{
            buf: [0u8; 78]
        }
    }

    pub fn encode<S: AsRef<[u8]>>(&mut self, input: S) -> anyhow::Result<&str> {
        let written = base64::prelude::BASE64_URL_SAFE_NO_PAD.encode_slice(input, &mut self.buf)?;
        if written != self.buf.len() {
            anyhow::bail!("Invalid encode buffer len!")
        }
        Ok(std::str::from_utf8(&self.buf).expect("Invalid UTF-8 in input!"))
    }
}


/// Pack both the Platform and the id from a match id into an u64
pub fn match_id_to_u64<S: AsRef<str>>(match_id: S) -> u64 {
    // split platform and id
    let mut split = match_id.as_ref().split('_');
    
    // platform to u64
    let platform = split.next().expect("Invalid match id!");
    let platform = riven::consts::PlatformRoute::from_str(platform).expect("Invalid match id!");
    let platform = platform as u64;
    
    // id to u64
    let id = split.next().expect("Invalid match id!");
    let mut id = id.parse::<u64>().expect("Invalid match id!");

    // pack the platform into the MSB of id
    id |= platform << (64-8);
    id
}

/// Unpack Platform and id from u64
pub fn u64_to_match_id(id_packed: u64) -> String {
    let mut match_id = String::with_capacity(20);
    
    // read platform id
    let platform = id_packed >> (64-8);
    let platform = riven::consts::PlatformRoute::try_from(platform as u8).expect("Invalid match id!");
    match_id.push_str(platform.into()); match_id.push('_');
    
    // read match id
    let id = (id_packed << 8) >> 8;
    match_id.push_str(&id.to_string());
    match_id
}


pub fn add_starting_handles(match_dbs: &RegionDB<heed::types::U64<heed::byteorder::NativeEndian>, heed::types::U8>, match_ids: &Vec<String>) -> anyhow::Result<()> {
    let invalid = "Invalid Starting Handle MatchId provided!";
    let mut wtxn = match_dbs.wtxn()?;
    
    for match_id in match_ids {
        // extract platform
        let mut split = match_id.split('_');
        let platform = split.next().expect(invalid);
        // cast platform
        let platform = riven::consts::PlatformRoute::from_str(platform).expect(invalid);
        
        // put platform and id
        match_dbs.get(&platform.to_regional()).expect("Starting Handle MatchId has Platform that is not included in the '--regions' Argument!")
            .get_or_put(&mut wtxn, &match_id_to_u64(match_id), &0)?;
        
        println!("Added {match_id} as Starting Handle.")
    }
    
    wtxn.commit()?;
    Ok(())
}
