use std::cmp::{min, max};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct SpanLoc {
    pub start_line: u32,
    pub start_column: u32,
    pub end_line: u32,
    pub end_column: u32,
}

impl SpanLoc {
    pub fn of(start_line: u32, start_column: u32, end_line: u32, end_column: u32) -> Self {
        SpanLoc {
            start_line, start_column, end_line, end_column,
        }
    }

    pub fn zero() -> Self {
        SpanLoc::of(0, 0, 0, 0)
    }

    pub fn merge(&mut self, other: SpanLoc) {
        let (sl, sc) = min((self.start_line, self.start_column), (other.start_line, other.start_column));
        let (el, ec) = max((self.end_line, self.end_column), (other.end_line, other.end_column));
        self.start_line = sl;
        self.start_column = sc;
        self.end_line = el;
        self.end_column = ec;
    }
}





