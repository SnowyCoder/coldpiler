use std::collections::BTreeSet;
use std::fmt::{Display, Error, Formatter};
use std::hash::Hash;

#[derive(Debug, PartialEq, Eq, Clone, Hash, PartialOrd, Ord)]
pub struct Partition (BTreeSet<u32>);

impl Partition {
    pub fn create_empty() -> Self {
        Partition(BTreeSet::new())
    }

    pub fn create_from(set: BTreeSet<u32>) -> Self {
        Partition(BTreeSet::from(set))
    }

    pub fn insert(&mut self, x: u32) {
        self.0.insert(x);
    }

    pub fn remove(&mut self, x: u32) {
        self.0.remove(&x);
    }

    pub fn contains(&self, x: u32) -> bool {
        self.0.contains(&x)
    }

    pub fn iter(&self) -> std::collections::btree_set::Iter<u32> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Display for Partition {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{:?}", self.0)
    }
}

