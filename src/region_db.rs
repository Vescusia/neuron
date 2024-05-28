use riven::consts::RegionalRoute;
use redb::{Database, Key, TableDefinition, Value};

use std::collections::HashMap;
use std::collections::hash_map::Values;


pub struct RegionDB<K: Key + 'static, V: Value + 'static> {
    pub db: Database,
    tables: HashMap<RegionalRoute, TableDefinition<'static, K, V>>
}

impl<K: Key, V: Value> RegionDB<K, V> {
    pub fn new<P: AsRef<str>>(path: P, regions: &[RegionalRoute]) -> anyhow::Result<Self> {
        // open db file
        let db = Database::create(path.as_ref())?;
        
        // create HashMap
        let mut tables = HashMap::new();
        
        // insert table defs
        for &region in regions {
            tables.insert(
                region,
                TableDefinition::new(region.into())
            );
        }
        
        // create tables
        let wtxn = db.begin_write()?;
        for &table in tables.values() {
            wtxn.open_table(table)?;
        }
        wtxn.commit()?;
        
        // fin~
        Ok(Self{
            db,
            tables,
        })
    }
    
    pub fn tables(&self) -> Values<RegionalRoute, TableDefinition<'_, K, V>> {
        self.tables.values()
    }
    
    pub fn read(&self) -> redb::Result<redb::ReadTransaction, redb::TransactionError> {
        self.db.begin_read()
    }
    
    pub fn write(&self) -> redb::Result<redb::WriteTransaction, redb::TransactionError> {
        self.db.begin_write()
    }
}


impl<K: Key, V: Value> std::ops::Index<RegionalRoute> for RegionDB<K, V> {
    type Output = TableDefinition<'static, K, V>;

    fn index(&self, index: RegionalRoute) -> &Self::Output {
        self.tables.get(&index).expect("Invalid RegionalRoute!")
    }
}
