pub mod radix_tree;
mod enumerable;

pub use enumerable::{Enumerable, ByteIterator};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
