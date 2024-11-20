use riven::models::match_v5::Participant;

pub struct ParticipantCombinations<'a>(pub &'a Vec<Participant>);

impl<'a> ParticipantCombinations<'a> {
    pub fn matchups(&'a self) -> Matchups<'a> {
        Matchups{ participants: self.0, x: 0, y: 0 }
    }
    
    pub fn synergies(&'a self) -> Synergies<'a> {
        Synergies{ participants: self.0, x: 0, y: 0 }
    }
}


/// Get all combinations of the Participants, where the participants are in opposite/different teams
pub struct Matchups<'a> {
    pub participants: &'a Vec<Participant>,
    x: usize,
    y: usize,
}

impl<'a> Iterator for Matchups<'a> {
    type Item = (&'a Participant, &'a Participant);

    fn next(&mut self) -> Option<Self::Item> {
        if self.y >= self.participants.len() {
            self.y = 0;
            self.x += 1;
        }
        let y_old = self.y; 
        self.y += 1;
        
        if self.x >= self.participants.len() {
            None 
        }
        else if self.participants[self.x].team_id == self.participants[y_old].team_id {
            self.next()
        } 
        else {
            Some((&self.participants[self.x], &self.participants[y_old]))
        }
    }
}


/// Get all combinations of the Participants, where the participants are in the same team (self-self combinations are included!)
pub struct Synergies<'a> {
    pub participants: &'a Vec<Participant>,
    x: usize,
    y: usize,
}

impl<'a> Iterator for Synergies<'a> {
    type Item = (&'a Participant, &'a Participant);

    fn next(&mut self) -> Option<Self::Item> {
        if self.y >= self.participants.len() {
            self.y = 0;
            self.x += 1;
        }
        let y_old = self.y; 
        self.y += 1;
        
        if self.x >= self.participants.len() {
            None 
        }
        else if self.participants[self.x].team_id != self.participants[y_old].team_id {
            self.next()
        } 
        else {
            Some((&self.participants[self.x], &self.participants[y_old]))
        }
    }
}
