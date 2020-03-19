use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::{DefaultHasher, RandomState};
use std::iter::FromIterator;

#[derive(Clone, Debug, Eq, Default)]
pub struct HashableSet<T: Hash + Eq + Clone>(pub HashSet<T>);

impl<T>  HashableSet<T> where T: Hash + Eq + Clone {
    pub fn new() -> Self {
        HashableSet(HashSet::new())
    }

    pub fn from(set: HashSet<T>) -> Self {
        HashableSet(set)
    }

    pub fn insert(&mut self, x: T) {
        self.0.insert(x);
    }

    pub fn remove(&mut self, x: &T) {
        self.0.remove(&x);
    }

    pub fn union(&mut self, other: &HashableSet<T>) {
        self.0.extend(other.iter().cloned())
    }

    pub fn contains(&self, x: &T) -> bool {
        self.0.contains(&x)
    }

    pub fn iter(&self) -> std::collections::hash_set::Iter<T> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<T: Hash + Eq + Clone> Hash for HashableSet<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut data = 0u64;
        for x in self.0.iter() {
            let mut s = DefaultHasher::new();
            x.hash(&mut s);
            data ^= s.finish();
        }
        state.write_usize(self.0.len());
        state.write_u64(data);
    }
}

impl<T: Hash + Eq + Clone> PartialEq for HashableSet<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T: Hash + Eq + Clone> From<HashSet<T>> for HashableSet<T> {
    fn from(x: HashSet<T, RandomState>) -> Self {
        HashableSet::from(x)
    }
}

impl<T: Hash + Eq + Clone> FromIterator<T> for HashableSet<T> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
        HashSet::from_iter(iter).into()
    }
}


