use std::iter::FusedIterator;

/// Creates an iterator from `from` to `to` with inclusive ends.
/// The call panics if the range is invalid (`to` < `from` or invalid char inside the range)
pub fn char_range_inclusive(from: char, to: char) -> CharRange {
    let from = from as u32;
    let to = to as u32;
    assert!(from < to);
    if from <= 0xDFFF && to >= 0xD800 {
        panic!("Invalid range!")
    }
    CharRange { from, to }
}

pub struct CharRange {
    from: u32,
    to: u32,
}

impl Iterator for CharRange {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        if self.from > self.to {
            return None
        }
        let x = self.from;
        self.from += 1;
        // SAFETY: assuming that the struct has been constructed from `char_range_inclusive`
        // then the boundaries were checked beforehand, the u32 s were constructed from chars and
        // only valid chars were between them.
        let ch = unsafe { std::char::from_u32_unchecked(x) };
        Some(ch)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let x = (self.to - self.from) as usize + 1;
        (x, Some(x))
    }
}

impl ExactSizeIterator for CharRange {
    fn len(&self) -> usize {
        (self.to - self.from) as usize + 1
    }
}

impl FusedIterator for CharRange {}

impl DoubleEndedIterator for CharRange {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.from > self.to {
            return None
        }
        let x = self.to;
        self.to -= 1;
        // SAFETY: assuming that the struct has been constructed from `char_range_inclusive`
        // then the boundaries were checked beforehand, the u32 s were constructed from chars and
        // only valid chars were between them.
        let ch = unsafe { std::char::from_u32_unchecked(x) };
        Some(ch)
    }
}
