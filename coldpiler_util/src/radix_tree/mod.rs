use std::cmp::Ordering;
use std::iter::FusedIterator;
use std::fmt::Debug;

type NodeIndex = usize;

pub trait Token : Clone + Copy + PartialOrd + Ord + PartialEq + Eq + Debug {}

impl<T> Token for T where T: Clone + Copy + PartialOrd + Ord + PartialEq + Eq + Debug {
}

// This is not the common Trie: the text is not on the edge but on the next node, this removes
// the need for a special root node and simplifies the code a little (I hope)

#[derive(Debug, Clone)]
struct Node<T: Token> {
    parent: NodeIndex,
    content: Vec<T>,
    leaf_id: Option<u32>,
    // The `T` in the edge is a lookahead
    children: Vec<(T, NodeIndex)>
}

impl<T: Token> Node<T> {
    fn find_next(&self, lookahead: T) -> Option<NodeIndex> {
        self.children.binary_search_by(|probe| {
            probe.0.cmp(&lookahead)
        })
            .ok()
            .map(|index| self.children[index].1)
    }

    fn insert_next(&mut self, lookahead: T, index: NodeIndex) -> bool {
        match self.children.binary_search_by(|probe| {
            probe.0.cmp(&lookahead)
        }) {
            Ok(_) => false,// key already found
            Err(pos) => {
                self.children.insert(pos, (lookahead, index));
                true
            },
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RadixTree<T: Token> {
    nodes: Vec<Node<T>>,
    id_to_node: Vec<NodeIndex>,
    next_id: u32,
}

impl<T: Token> RadixTree<T> {
    pub fn new() -> Self {
        RadixTree {
            nodes: Vec::new(),
            id_to_node: Vec::new(),
            next_id: 0
        }
    }

    /// Searches the nodes for a key match, if found returns the match index
    /// and a bool, the bool will be false only when the match is partial.
    fn get_index(&self, mut key: &[T]) -> Option<(NodeIndex, bool)> {
        if self.nodes.is_empty() {
            return None// Nothing to match
        }
        let mut current_node_index = 0;

        loop {
            let current_node = &self.nodes[current_node_index];
            let matches_until = string_match(key, current_node.content.as_slice());

            let key_cmp = matches_until.cmp(&key.len());
            let node_cmp = matches_until.cmp(&current_node.content.len());

            match (key_cmp, node_cmp) {
                (Ordering::Equal, Ordering::Equal) => {
                    // Perfect match
                    return Some((current_node_index, true))
                },
                (Ordering::Less, Ordering::Less) => {
                    // Search key and node content don't match
                    return None
                }
                (Ordering::Equal, Ordering::Less) => {
                    // Partial match, the search key matches but the node content is longer
                    return Some((current_node_index, false))
                }
                (Ordering::Less, Ordering::Equal) => {
                    // The search key matches but it's longer than the current node
                    // (code continues after match)
                },
                _ => unreachable!()// We can't have a "Greater" match.
            }

            key = &key[current_node.content.len()..];
            // Search for next node
            // Can the index operation fail? no: we already compared the key and the node content
            // and we saw that  the key is longer, this means that there is at least one element
            // after we skip content.len() elements.
            let lookahead = key[0];
            current_node_index = match current_node.find_next(lookahead) {
                None => return None,// No edge matched, research failed.
                Some(x) => x,
            };
        }
    }

    pub fn get<R: AsRef<[T]>>(&self, key: R) -> Option<u32> {
        let key = key.as_ref();
        match self.get_index(key) {
            None => None, // No match
            Some((_, false)) => None,// Partial match
            Some((index, true)) => self.nodes[index].leaf_id // Full match
        }
    }

    fn create_node(&mut self, parent: NodeIndex, content: Vec<T>, leaf_id: Option<u32>) -> NodeIndex {
        let index = self.nodes.len();
        self.nodes.push(Node {
            parent, content, leaf_id,
            children: Vec::new(),
        });
        if let Some(id) = leaf_id {
            self.id_to_node[id as usize] = index;
        }
        index
    }

    fn split_node(&mut self, node_index: NodeIndex, after_len: usize) {
        let new_node_index = self.nodes.len();
        // Split the content of the node
        let new_content = self.nodes[node_index].content.split_off(after_len);
        let new_lookahead = new_content[0];

        // TODO: somehow remove clone...
        for (_lookahead, ch_id) in &self.nodes[node_index].children.clone() {
            self.nodes[*ch_id].parent = new_node_index;
        }

        // Create the new node with the `children` vector that will be placed in the original node
        // this lets us simplify the movement of the children with a fast memory swap.
        // This step is quite complicated so I'll explain it:
        // We are creating a new node that will be after the original node
        // The next node will have the old leaf_id (or value) and children
        // We also create an edge from the original node and the new node (with the appropriate lookahead).
        self.nodes.push(Node {
            parent: node_index,
            content: new_content,
            leaf_id: self.nodes[node_index].leaf_id,
            children: vec![(new_lookahead, new_node_index)],
        });
        // Little trick to index an array two times mutably:
        let (first_half, last) = self.nodes.split_at_mut(new_node_index);
        let new_node = &mut last[0];
        let original_node = &mut first_half[node_index];

        std::mem::swap(&mut original_node.children, &mut new_node.children);
        original_node.leaf_id = None;
        // Update id_to_node
        if let Some(id) = new_node.leaf_id {
            self.id_to_node[id as usize] = new_node_index;
        }
        // Reset the new_node's parent link.
        /*for (_lookahead, ch_id) in &new_node.children {
            self.nodes[*ch_id].parent = new_node_index;
        }*/
    }

    fn node_insert(&mut self, current_node_index: NodeIndex, key: &[T], id: u32) -> u32 {
        let mut current_node = &mut self.nodes[current_node_index];
        let matches_until = string_match(&current_node.content, key);

        // 3 cases
        // - key == current_node.content => the node is this, no insertion required
        // - key >= current_node.content => pass the burden to another node
        // - key <= current_node.content => split this node at the match position, then add another
        //          node with the new key

        let key_cmp = matches_until.cmp(&key.len());
        let node_cmp = matches_until.cmp(&current_node.content.len());

        match (key_cmp, node_cmp) {
            (Ordering::Equal, Ordering::Equal) => {
                // the key matches this node's content: this is the node we're searching for
                if current_node.leaf_id.is_none() {
                    current_node.leaf_id = Some(id);
                    self.id_to_node[id as usize] = current_node_index;
                    id
                } else {
                    current_node.leaf_id.unwrap()
                }
            },
            (Ordering::Less, Ordering::Equal) => {
                // The key matches this node's content but it's longer
                let lookahead = key[matches_until];
                match current_node.find_next(lookahead) {
                    None => {
                        let new_node = self.create_node(current_node_index, key[matches_until..].to_vec(), Some(id));
                        // If I use current_node here rustc screams
                        self.nodes[current_node_index].insert_next(lookahead, new_node);
                        id
                    },
                    Some(index) => {
                        self.node_insert(index, &key[matches_until..], id)
                    },
                }
            },
            (Ordering::Equal, Ordering::Less) => {
                // Split this node and assign the first part to the new id
                self.split_node(current_node_index, matches_until);
                self.nodes[current_node_index].leaf_id = Some(id);
                id
            },
            (Ordering::Less, Ordering::Less) => {
                // Split this node and create a new leaf to store the remains of the key
                self.split_node(current_node_index, matches_until);
                // We have just splitted and we know that the only child of the new node has a
                // lookahead that doesn't match our key, so it is safe to insert without previously
                // checking if the child already exists.
                let new_node_index = self.create_node(current_node_index, key[matches_until..].to_vec(), Some(id));
                // If I use current_node here rustc screams
                self.nodes[current_node_index].insert_next(key[matches_until], new_node_index);
                id
            }
            _ => unreachable!()
        }
    }

    pub fn insert<R: AsRef<[T]>>(&mut self, key: R) -> u32 {
        let key = key.as_ref();
        if self.nodes.is_empty() {
            let id = self.next_id;
            self.next_id += 1;
            self.id_to_node.push(0);
            self.create_node(0, key.to_vec(), Some(id));
            id
        } else {
            let allocated_id = self.next_id;
            self.id_to_node.push(0);

            let real_id = self.node_insert(0, key, allocated_id);
            if allocated_id == real_id {
                self.next_id += 1;
            } else {
                // Deallocate
                self.id_to_node.pop();
            }
            real_id
        }
    }

    pub fn unravel_node_ref(&self, node_id: NodeIndex) -> Vec<&[T]> {
        let mut current_id = node_id;
        let mut stack = Vec::new();

        loop {
            let current = &self.nodes[current_id];
            stack.push(current.content.as_slice());
            if current.parent == current_id {
                // Root
                break
            }
            current_id = current.parent;
        }
        stack.reverse();
        stack
    }

    pub fn unravel_node(&self, node_id: NodeIndex) -> Vec<T> {
        let first_token = match self.nodes[node_id].content.get(0) {
            None => {
                // Empty path
                return Vec::new()
            },
            Some(token) => token,
        };

        let mut length = 0;
        let mut current_id = node_id;

        loop {
            let current = &self.nodes[current_id];
            length += current.content.len();
            if current.parent == current_id {
                // Root
                break;
            }
            current_id = current.parent;
        }

        current_id = node_id;

        let mut str = Vec::new();
        str.resize(length, *first_token);
        let mut str_slice = str.as_mut_slice();

        loop {
            let current = &self.nodes[current_id];
            let (ls, curr_slice) = str_slice.split_at_mut(str_slice.len() - current.content.len());
            str_slice = ls;
            curr_slice.copy_from_slice(&current.content.as_slice());
            if current.parent == current_id {
                // Root
                break;
            }
            current_id = current.parent;
        }

        str
    }

    pub fn find_key(&self, value: u32) -> Vec<T> {
        self.unravel_node(self.id_to_node[value as usize])
    }

    pub fn get_prefix<S: AsRef<[T]>>(&self, prefix: S) -> DFSIterator<T> {
        let prefix = prefix.as_ref();
        match self.get_index(prefix) {
            None => {
                DFSIterator {
                    tree: self,
                    next_node: 0,
                    last_node: 0,
                    done: true
                }
            },
            Some((node, _complete)) => {
                DFSIterator {
                    tree: self,
                    next_node: node,
                    last_node: node,
                    done: false
                }
            },
        }
    }

    fn next_sibling(&self, node_index: NodeIndex) -> Option<NodeIndex> {
        if node_index == 0 {
            return None// Root
        }
        let node = &self.nodes[node_index];
        let parent = &self.nodes[node.parent];
        let node_lookahead = node.content[0];

        let pos = parent.children.binary_search_by(|probe| {
            probe.0.cmp(&node_lookahead)
        }).ok().unwrap();// The children is registered, so it will always find it.

        parent.children.get(pos + 1).map(|(_lookahead, node_index)| *node_index)
    }
}

#[derive(Clone)]
pub struct DFSIterator<'a, T: Token> {
    tree: &'a RadixTree<T>,
    next_node: NodeIndex,
    last_node: NodeIndex,
    done: bool
}

impl<'a, T: Token> DFSIterator<'a, T> {
    fn next_sibling_rec(&self, mut node_index: NodeIndex) -> Option<NodeIndex> {
        loop {
            if node_index == self.last_node {
                return None
            }
            match self.tree.next_sibling(node_index) {
                Some(x) => {
                    return Some(x);
                },
                None => {
                    node_index = self.tree.nodes[node_index].parent;
                },
            }
        }
    }

    fn advance_node(&mut self) -> bool {
        if self.done {
            return false
        }
        // Check first child
        let next_node = &self.tree.nodes[self.next_node];
        if let Some(child) = next_node.children.get(0) {
            self.next_node = child.1;
            return true
        }
        // No child found: check sibling
        match self.next_sibling_rec(self.next_node) {
            Some(sibling) => {
                self.next_node = sibling;
                true
            },
            None => {
                self.done = true;
                false
            },
        }
    }
}

impl<'a, T: Token> Iterator for DFSIterator<'a, T> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        while self.tree.nodes[self.next_node].leaf_id.is_none() {
            if !self.advance_node() {
                return None
            }
        }

        let ret = self.tree.nodes[self.next_node].leaf_id.unwrap();
        self.advance_node();

        Some(ret)
    }
}

impl<'a, T: Token> FusedIterator for DFSIterator<'a, T> {
}

/// Searches the index where the two strings do not match anymore.
/// If the two strings do not match at all then 0 is returned.
///
/// # Examples
///
/// ```
/// # use coldpiler_util::radix_tree::string_match;
/// assert_eq!(string_match("romulus", "roman"), 3);
/// assert_eq!(string_match("romulus", "ro"), 2);
/// assert_eq!(string_match("ro", "roman"), 2);
/// assert_eq!(string_match("banana", "roman"), 0);
/// assert_eq!(string_match("banana", "banana"), 6);
/// ```
pub fn string_match<T: PartialEq, R: AsRef<[T]>, S: AsRef<[T]>>(a: R, b: S) -> usize {
    let a = a.as_ref();
    let b = b.as_ref();
    let mut matches_until = 0;
    while let Some(e) = a.get(matches_until) {
        if Some(e) == b.get(matches_until) {
            matches_until += 1;
        } else {
            break
        }
    }
    matches_until
}


#[cfg(test)]
mod tests {
    use crate::radix_tree::RadixTree;

    #[test]
    fn basic() {
        let mut tree = RadixTree::new();

        assert_eq!(tree.get("hello"), None);
        let hi = tree.insert("hi");
        assert_eq!(tree.get("h"), None);
        assert_eq!(tree.get("hi"), Some(hi));
        assert_eq!(tree.get("hit"), None);
        let hello = tree.insert("hello");
        assert_eq!(tree.get("h"), None);
        assert_eq!(tree.get("hi"), Some(hi));
        assert_eq!(tree.get("hell"), None);
        assert_eq!(tree.get("hello"), Some(hello));
        let hellish = tree.insert("hellish");
        assert_eq!(tree.get("hell"), None);
        assert_eq!(tree.get("hello"), Some(hello));
        assert_eq!(tree.get("hellish"), Some(hellish));
        assert_eq!(tree.get("helli"), None);
        let hell = tree.insert("hell");
        assert_eq!(tree.get("hello"), Some(hello));
        assert_eq!(tree.get("hell"), Some(hell));
        assert_eq!(tree.get("hellish"), Some(hellish));
        assert_eq!(tree.insert("hello"), hello);

        // I know it's misspelled
        let hellp = tree.insert("hellp");
        let hello_world = tree.insert("hello world");

        let mut iter = tree.get_prefix("hel");
        assert_eq!(iter.next(), Some(hell));
        assert_eq!(iter.next(), Some(hellish));
        assert_eq!(iter.next(), Some(hello));
        assert_eq!(iter.next(), Some(hello_world));
        assert_eq!(iter.next(), Some(hellp));
        assert_eq!(iter.next(), None);

        let unr = tree.find_key(hello_world);
        assert_eq!(unr.as_slice(), b"hello world");
    }

    #[test]
    fn basic01() {
        let mut tree = RadixTree::new();
        tree.insert("not");
        tree.insert("n");
        // Split the node without creating a third one (yes this was a bug)
    }
}




