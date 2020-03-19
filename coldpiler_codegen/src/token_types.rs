use coldpiler_parser::parser::Enumerable;
use std::sync::atomic::Ordering;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::sync::atomic::AtomicU32;

static SCANNER_PLACEHOLDER_TYPE_LEN: AtomicU32 = AtomicU32::new(32);
static PARSER_PLACEHOLDER_TYPE_LEN: AtomicU32 = AtomicU32::new(32);


// TODO: use Macros v1 to duplicate code
macro_rules! create_parser_type {
    ($name:ident, $atomic:ident) => {
        #[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
        pub struct $name(pub u32);

        impl $name {
            pub fn get_size() -> u32 {
                $atomic.load(Ordering::SeqCst)
            }

            pub fn set_size(new_size: u32) {
                $atomic.store(new_size, Ordering::SeqCst);
            }
        }

        impl InstanceIndexable for $name {
            fn instance(i: usize) -> Self {
                $name(i as u32)
            }
        }

        impl Enumerable for $name {
            type Iterator = PlaceholderTypeIterator<Self>;

            fn index(&self) -> usize {
                self.0 as usize
            }

            fn enumerate() -> Self::Iterator {
                let max_len: u32 = Self::get_size();
                PlaceholderTypeIterator {
                    next: 0,
                    size: max_len as usize,
                    ph: PhantomData,
                }
            }
        }
    }
}

create_parser_type!(ScannerPlaceholderType, SCANNER_PLACEHOLDER_TYPE_LEN);
create_parser_type!(ParserPlaceholderType, PARSER_PLACEHOLDER_TYPE_LEN);

pub trait InstanceIndexable {
    fn instance(i: usize) -> Self;
}

pub struct PlaceholderTypeIterator<T: InstanceIndexable> {
    next: usize,
    size: usize,
    ph: PhantomData<T>,
}

impl<T: InstanceIndexable> Iterator for PlaceholderTypeIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.size {
            return None;
        }
        let curr = self.next;
        self.next += 1;
        Some(T::instance(curr))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.size - self.next;
        (size, Some(size))
    }

    fn nth(&mut self, index: usize) -> Option<Self::Item> {
        let add_res = self.next.checked_add(index);
        let add_res = match add_res {
            Some(x) => x,
            None => {
                self.next = self.size;
                return None
            }
        };
        if add_res >= self.size {
            self.next = self.size;
            None
        } else {
            self.next += index;
            self.next()
        }
    }
}

impl<T: InstanceIndexable> ExactSizeIterator for PlaceholderTypeIterator<T> {
    fn len(&self) -> usize {
        self.size - self.next
    }
}

impl<T: InstanceIndexable> FusedIterator for PlaceholderTypeIterator<T> {}
