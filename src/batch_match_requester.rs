use riven::consts::RegionalRoute;
use riven::models::match_v5::Match;
use riven::RiotApi;
use tokio::sync::mpsc;
use crate::encoded_match_id::EncodedMatchId;


#[derive(Debug)]
pub struct BatchMatchRequester {
    pub responses: Vec<(riven::Result<Option<Match>>, EncodedMatchId)>,
    senders: Vec<mpsc::Sender<EncodedMatchId>>,
    funnel: mpsc::Receiver<(riven::Result<Option<Match>>, EncodedMatchId)>,
}

impl<'a> BatchMatchRequester {
    pub fn new(thread_amount: usize, api: std::sync::Arc<RiotApi>, region: RegionalRoute) -> Self {
        // create receive channel
        let (tx, funnel) = mpsc::channel(thread_amount);
        
        let mut senders = Vec::with_capacity(thread_amount);
        
        // spawn threads
        for _ in 0..thread_amount {
            // create send channel for each thread
            let (sender, rx) = mpsc::channel(1);
            senders.push(sender);
            // clone arcs
            let tx = tx.clone();
            let api = api.clone();
            // spawn thread
            tokio::task::spawn(async move { request_thread(rx, tx, api, region).await });
        }
        
        Self{
            responses: Vec::with_capacity(thread_amount),
            senders,
            funnel
        }
    }
    
    /// Fill the response buffer with Requests, with a timeout
    /// 
    /// This returns [`None`] when `requests` is empty.
    /// 
    /// Also, this means that the returned slice can explicitly be of length 0!
    pub async fn request<I: Iterator<Item = EncodedMatchId> + Sized>(&'a mut self, requests: I) -> Option<&'a [(riven::Result<Option<Match>>, EncodedMatchId)]> {
        // clear now outdated responses
        self.responses.clear();
        
        // send requests to threads
        let mut sent = 0;
        for (sender, request) in self.senders.iter().zip(requests) {
            sender.send(request).await.unwrap();
            sent += 1;
        }
        
        // check if requests was empty
        if sent == 0 {
            return None
        }
        
        // receive responses from threads
        for _ in 0..sent {
            self.responses.push(self.funnel.recv().await.unwrap())
        }
        
        Some(self.responses.as_slice())
    }
}


async fn request_thread(mut rx: mpsc::Receiver<EncodedMatchId>, tx: mpsc::Sender<(riven::Result<Option<Match>>, EncodedMatchId)>, api: std::sync::Arc<RiotApi>, region: RegionalRoute) {
    // wrap arguments
    let match_v5 = api.match_v5();

    // main request loop
    while let Some(match_id) = rx.recv().await {
        tx.send((
            match_v5.get_match(
                region,
                &match_id.to_string()
            ).await,
            match_id
        )).await.unwrap()
    }
}
