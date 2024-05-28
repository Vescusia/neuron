use riven::consts::RegionalRoute;
use riven::models::match_v5::Match;
use riven::RiotApi;
use tokio::sync::mpsc;
use tokio::time::timeout;
use crate::encoded_match_id::EncodedMatchId;


#[derive(Debug)]
pub struct BatchMatchRequester {
    pub responses: Vec<(riven::Result<Option<Match>>, EncodedMatchId)>,
    senders: Vec<(mpsc::Sender<EncodedMatchId>, bool)>,
    funnel: mpsc::Receiver<(riven::Result<Option<Match>>, EncodedMatchId, usize)>,
}

impl<'a> BatchMatchRequester {
    pub fn new(thread_amount: usize, api: std::sync::Arc<RiotApi>, region: RegionalRoute) -> Self {
        // create receive channel
        let (tx, funnel) = mpsc::channel(thread_amount);
        
        let mut senders = Vec::with_capacity(thread_amount);
        
        // spawn threads
        for t in 0..thread_amount {
            // create send channel for each thread
            let (sender, rx) = mpsc::channel(1);
            senders.push((sender, true));
            // clone arcs
            let tx = tx.clone();
            let api = api.clone();
            // spawn thread
            tokio::task::spawn(async move { request_thread(rx, tx, api, region, t).await });
        }
        
        Self{
            responses: Vec::with_capacity(thread_amount),
            senders,
            funnel
        }
    }
    
    /// Fill the response buffer with Requests, with a timeout
    /// 
    /// This returns [`None`] when the amount of requests in `requests` 
    /// is less than the amount of idle request Threads.
    /// (Threads will only become idle again, when a [`request`] call returns [`Some`]!)
    /// 
    /// In other words, this returns [`None`] when the Iterator "ran out".
    /// 
    /// Also, this means that the returned slice can explicitly be of length 0!
    pub async fn request<I: Iterator<Item = EncodedMatchId>>(&'a mut self, mut requests: I, wait_time: std::time::Duration) -> Option<&'a [(riven::Result<Option<Match>>, EncodedMatchId)]> {
        // clear now outdated responses
        self.responses.clear();
        
        // filter idle threads
        let threads = self.senders
            .iter_mut()
            .filter(|(_, is_idle)| *is_idle);
        
        // send requests to threads
        for (sender, is_idle) in threads {
            if let Some(request) = requests.next() {
                *is_idle = false;
                sender.send(request).await.unwrap();
            }
            // if the requests Iterator finished before exhausting all threads,
            // return None.
            else {
                return None
            }
        }
        
        // receive responses from threads
        let _ = timeout(wait_time, async {
            loop {
                let (response, id, t_id) = self.funnel.recv().await.unwrap();
                self.responses.push((response, id));
                // set back to idle
                self.senders[t_id].1 = true;
            }
        }).await;
        
        //println!("{} of {}", self.responses.len(), self.senders.iter().filter(|(_, idle)| *idle).count());
        Some(self.responses.as_slice())
    }
}


async fn request_thread(mut rx: mpsc::Receiver<EncodedMatchId>, tx: mpsc::Sender<(riven::Result<Option<Match>>, EncodedMatchId, usize)>, api: std::sync::Arc<RiotApi>, region: RegionalRoute, thread: usize) {
    // wrap arguments
    let match_v5 = api.match_v5();

    // main request loop
    while let Some(match_id) = rx.recv().await {
        tx.send((
            match_v5.get_match(
                region,
                &match_id.to_string()
            ).await,
            match_id,
            thread
        )).await.unwrap()
    }
}
