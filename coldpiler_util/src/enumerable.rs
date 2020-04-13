use std::iter::FusedIterator;

pub trait Enumerable : Sized + Copy {
    type Iterator: Iterator<Item = Self> + std::iter::ExactSizeIterator;

    fn index(&self) -> usize;

    fn enumerate() -> Self::Iterator;
}


impl Enumerable for u8 {
    type Iterator = ByteIterator;

    fn index(&self) -> usize {
        *self as usize
    }

    fn enumerate() -> Self::Iterator {
        ByteIterator {
            next: 0u16
        }
    }
}

pub struct ByteIterator {
    next: u16,
}

impl Iterator for ByteIterator {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next < u8::max_value() as u16 {
            let n = self.next;
            self.next += 1;
            Some(n as u8)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = (u8::max_value() as u16 - self.next + 1) as usize;
        (size, Some(size))
    }
}

impl FusedIterator for ByteIterator {
}

impl ExactSizeIterator for ByteIterator {
    fn len(&self) -> usize {
        (u8::max_value() as u16 - self.next + 1) as usize
    }
}