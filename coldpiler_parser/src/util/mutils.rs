
// mutable utils (mutils)


// Thanks bluss https://stackoverflow.com/questions/30073684/how-to-get-mutable-references-to-two-array-elements-at-the-same-time
pub enum IndexTwice<T> {
    Both(T, T),
    One(T),
    None,
}

pub fn index_twice<T>(slc: &mut [T], a: usize, b: usize) -> IndexTwice<&mut T> {
    if a == b {
        slc.get_mut(a).map_or(IndexTwice::None, IndexTwice::One)
    } else if a >= slc.len() || b >= slc.len() {
        IndexTwice::None
    } else {
        // safe because a, b are in bounds and distinct
        unsafe {
            let ar = &mut *(slc.get_unchecked_mut(a) as *mut _);
            let br = &mut *(slc.get_unchecked_mut(b) as *mut _);
            IndexTwice::Both(ar, br)
        }
    }
}