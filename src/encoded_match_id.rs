use std::fmt::Display;
use std::str::FromStr;

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct EncodedMatchId(pub u64);

impl EncodedMatchId {
    const PLATFORM_MASK: u64 = 0xFF00000000000000;
    const PLATFORM_SHIFT: u8 = 56;
    const ID_MASK: u64 = !Self::PLATFORM_MASK;
    
    pub fn id(self) -> u64 {
        self.0 & Self::ID_MASK
    }
    
    pub fn platform(self) -> riven::consts::PlatformRoute {
        let platform = (self.0 & Self::PLATFORM_MASK) >> Self::PLATFORM_SHIFT;
        riven::consts::PlatformRoute::try_from(platform as u8).expect("Invalid PlatformId encoded!")
    }
}



impl<S: AsRef<str>> From<S> for EncodedMatchId {
    fn from(match_id: S) -> Self {
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
        id |= platform << Self::PLATFORM_SHIFT;
        Self(id)
    }
}

impl From<&EncodedMatchId> for String {
    fn from(encoded_id: &EncodedMatchId) -> Self {
        let mut match_id = String::with_capacity(20);
        
        // read platform id
        match_id.push_str(encoded_id.platform().into()); match_id.push('_');
        
        // read match id
        match_id.push_str(&encoded_id.id().to_string());
        match_id
    }
}

impl Display for EncodedMatchId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", String::from(self))
    }
}

impl From<EncodedMatchId> for u64 {
    fn from(value: EncodedMatchId) -> Self {
        value.0
    }
}

impl std::borrow::Borrow<u64> for EncodedMatchId {
    fn borrow(&self) -> &u64 {
        &self.0
    }
}
