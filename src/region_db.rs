use std::collections::hash_map::Values;
use heed::Database;
use riven::consts::RegionalRoute;

use std::collections::HashMap;


#[derive(Debug)]
pub struct RegionDB<KC: 'static, DC: 'static> {
    pub env: heed::Env,
    dbs: HashMap<RegionalRoute, Database<KC, DC>>
}

impl<KC, DC> RegionDB<KC, DC> {
    pub fn new<P: AsRef<str>>(path: P, regions: &[RegionalRoute]) -> anyhow::Result<Self> {
        std::fs::create_dir_all(path.as_ref())?;

        // open db file
        let env = unsafe {
            heed::EnvOpenOptions::new()
                .max_dbs(16)
                .open(path.as_ref())
        }?;
        let mut wtxn = env.write_txn()?;
        
        // create HashMap
        let mut dbs = HashMap::new();
        
        // create dbs
        for region in regions {
            dbs.insert(
                *region,
                env.create_database(&mut wtxn, Some(&region.to_string()))?
            );
        }
        wtxn.commit()?;
        
        Ok(Self{
            env,
            dbs
        })
    }
    
    pub fn get(&self, region: &RegionalRoute) -> Option<&Database<KC, DC>> {
        self.dbs.get(region)
    }
    
    pub fn dbs(&self) -> Values<RegionalRoute, Database<KC, DC>> {
        self.dbs.values()
    }
    
    pub fn wtxn(&self) -> heed::Result<heed::RwTxn> {
        self.env.write_txn()
    }
    
    pub fn rtxn(&self) -> heed::Result<heed::RoTxn> {
        self.env.read_txn()
    }
}
