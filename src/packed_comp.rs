include!(concat!(env!("OUT_DIR"), "/champ_to_u8.rs"));


#[derive(Copy, Clone)]
pub struct PackedComp {
    pub packed: u64,
    pub ord: std::cmp::Ordering
}


impl PackedComp {
    /// Pack two teams into an u64
    /// If any [`Champion`] is [`Champion::NONE`], this will panic!
    ///
    /// Team order does NOT matter, it is handled internally!
    pub fn pack(team1: [Champion; 5], team2: [Champion; 5]) -> Self {
        Self::pack_ordered(team1, team2, Self::cmp_comps_u8(team1, team2))
    }

    /// Pack two teams into an u64.
    /// If any [`Champion`] is [`Champion::NONE`], this will panic!
    ///
    /// `ord` has to be relative to the **first** Team passed!!!
    ///
    /// (i.e. `team1` < `team2` => `pack_ordered(team1, team2, std::cmp::Less)` or `pack_ordered(team2, team1, std::cmp::Greater)`)
    pub fn pack_ordered(team1: [Champion; 5], team2: [Champion; 5], ord: std::cmp::Ordering) -> Self {
        if ord == std::cmp::Ordering::Less {
            Self::pack_teams_u64(
                team2.map(champ_to_u8),
                team1.map(champ_to_u8),
                ord
            )
        } else {
            Self::pack_teams_u64(
                team1.map(champ_to_u8),
                team2.map(champ_to_u8),
                ord
            )
        }
    }

    /// Packs two teams deterministically into an u64.
    ///
    /// The teams have to be ordered in a deterministic manner externally!
    fn pack_teams_u64(mut team1: [u8; 5], mut team2: [u8; 5], ord: std::cmp::Ordering) -> Self {
        // sort both team comps to remove importance of draft order
        team1.sort_unstable(); team2.sort_unstable();

        // work in the champs
        let mut packed = 0;
        for champ in team1.into_iter().chain(team2.into_iter()) {
            packed ^= champ as u64;
            packed = packed.rotate_left(7);
            //println!("{packed:064b}, {:08b}", champ)
        }

        Self {
            packed,
            ord
        }
    }

    fn cmp_comps_u8(team1: [Champion; 5], team2: [Champion; 5]) -> std::cmp::Ordering {
        let t1_value: u16 = team1.map(|c| c.0 as u16).into_iter().sum();
        let t2_value = team2.map(|c| c.0 as u16).into_iter().sum();
        t1_value.cmp(&t2_value)
    }
}


impl PartialEq for PackedComp {
    fn eq(&self, other: &Self) -> bool {
        self.packed == other.packed
    }
}
impl Eq for PackedComp {}

impl std::fmt::Debug for PackedComp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}", self.packed)
    }
}


#[cfg(test)]
mod tests {
    use rand::prelude::SliceRandom;
    use rand::Rng;
    use riven::consts::Champion;
    use super::*;


    #[test]
    fn test_pack_champs() {
        // init teams
        let mut team1 = [
            Champion::URGOT,
            Champion::YASUO,
            Champion::YUUMI,
            Champion::ZERI,
            Champion::DIANA
        ];
        let mut team2 = [
            Champion::FIORA,
            Champion::AURELION_SOL,
            Champion::TWITCH,
            Champion::BRAUM,
            Champion::LEE_SIN
        ];

        // initial value
        let start = PackedComp::pack(team1, team2);

        // start shuffling
        let mut rng = rand::thread_rng();
        for i in 0..100_000 {
            team1.shuffle(&mut rng);
            team2.shuffle(&mut rng);

            if i % 2 == 0 {
                assert_eq!(start, PackedComp::pack(team1, team2))
            } else {
                assert_eq!(start, PackedComp::pack(team2, team1))
            }
        }

        // start random team testing
        let mut all_known = Champion::ALL_KNOWN;
        all_known.sort_unstable();
        for _ in 0..100_000 {
            let (rand_team1, rand_team2) = gen_rand_teams(&mut rng);
            let packed = PackedComp::pack(rand_team1.map(|c| all_known[c as usize]), rand_team2.map(|c| all_known[c as usize]));

            assert_ne!(packed, start, "Packed Collision (could be coincidence):\n{team1:?} vs. {team2:?}\n{rand_team1:?} vs. {rand_team2:?}");
        }
    }

    fn gen_rand_teams(rng: &mut rand::rngs::ThreadRng) -> ([u8; 5], [u8; 5]) {
        // generate
        let mut teams = [0; 10];
        for i in 0..teams.len() {
            let num = loop {
                let num: u8 = rng.gen_range(0..=166);
                if !teams.iter().any(|c| *c == num) {
                    break num
                }
            };
            teams[i] = num
        }
        // extract
        let (team1, team2) = teams.split_first_chunk::<5>().unwrap();
        let (team2, _) = team2.split_first_chunk::<5>().unwrap();
        (*team1, *team2)
    }
}