use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Error, Formatter};

use crate::util::Partition;
use super::scanner::{CustomTokenType, Scanner};

#[derive(Clone, Debug, Default)]
pub struct NonDeterministicFiniteAutomaton<T: CustomTokenType> {
    nodes: Vec<Option<T>>,
    neighbours : Vec<Vec<(u32, Option<u8>)>>,
}

pub type NFA<T> = NonDeterministicFiniteAutomaton<T>;

impl<T: CustomTokenType> NonDeterministicFiniteAutomaton<T> {
    pub fn new() -> Self {
        NonDeterministicFiniteAutomaton {
            nodes: vec![],
            neighbours: vec![],
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn add_node(&mut self, token: Option<T>) -> u32 {
        self.nodes.push(token);
        self.neighbours.push(vec![]);
        self.nodes.len() as u32 - 1
    }

    pub fn add_empty_nodes(&mut self, len: usize) {
        let new_len = self.nodes.len() + len;
        self.nodes.resize(new_len, None);
        self.neighbours.resize(new_len, vec![]);
    }

    pub fn reserve_nodes(&mut self, additional: usize) {
        self.nodes.reserve(additional);
        self.neighbours.reserve(additional);
    }

    pub fn set_node_token(&mut self, index: u32, token: Option<T>) {
        self.nodes[index as usize] = token;
    }

    pub fn get_node_token(&self, index: u32) -> Option<T> {
        self.nodes[index as usize]
    }

    pub fn add_edge(&mut self, from: u32, to: u32, ch: Option<u8>) {
        // TODO: check?
        assert!(to < self.node_count() as u32);
        self.neighbours[from as usize].push((to, ch))
    }

    pub fn get_edges(&self, node: u32) -> &[(u32, Option<u8>)] {
        &self.neighbours[node as usize]
    }

    pub fn add_nfa(&mut self, other: &Self) -> u32 {
        let start_id = self.nodes.len() as u32;

        for x in other.nodes.iter() {
           self.add_node(*x);
        }

        for (from_node, neighbours) in other.neighbours.iter().enumerate() {
            for (to_node, ch) in neighbours {
                self.add_edge(start_id + from_node as u32, start_id + *to_node, *ch);
            }
        }

        start_id
    }

    pub fn get_alphabet(&self) -> HashSet<u8> {
        self.neighbours.iter().flat_map(|x| x.iter().filter_map(|(_, y)| *y)).collect()
    }

    pub fn map_tokens<F, N>(self, f: F) -> NFA<N>
        where F: Fn(Option<T>) -> Option<N>, N: CustomTokenType {
        NFA {
            nodes: self.nodes.iter().map(|&x| f(x)).collect(),
            neighbours: self.neighbours,
        }
    }

    pub fn to_dfa_direct(&self) -> Option<Scanner<T>> {
        let mut res = Scanner::new(self.nodes.len() as u32);
        for (i, x) in self.nodes.iter().enumerate() {
           if let Some(token) = x {
               res.set_token(i as u32, *token);
           }
        }
        for (from, x) in self.neighbours.iter().enumerate() {
            for (to, ch) in x {
                if let Some(ch) = ch {
                    // Check if another edge exists from node "from" and character "ch"
                    // in that case the NFA doesn't represent directly a DFA, so fail.
                    if res.get_next(from as u32, *ch as u8) != None {
                        return None;
                    }
                    res.add_edge(from as u32, *to, *ch);
                }
            }
        }
        Some(res)
    }

    pub fn to_dfa(&self) -> Result<Scanner<T>, NfaToDfaError<T>> {
        let mut res = NFA::new();

        /**
         * Follows the no-char paths from the nodes
         *   :param partition: The initial partition
         *   :return: The initial partition with their no-char successors added
         */
        fn no_char_closure<T: CustomTokenType>(nfa: &NFA<T>, partition: &mut Partition) {
            let mut work: Vec<u32> = partition.iter().copied().collect();

            while let Some(node) = work.pop() {
                for (next_node, char) in nfa.get_edges(node) {
                    if *char == None && !partition.contains(*next_node) {
                        partition.insert(*next_node);
                        work.push(*next_node);
                    }
                }
            }
        }

        /**
         * Follows the no-char paths from the nodes
         *   :param partition: The initial partition
         *   :return: The initial partition with their no-char successors added
         */
        fn advance_partition<T: CustomTokenType>(nfa: &NFA<T>, partition: &Partition, ch: u8) -> Partition {
            let mut res = Partition::create_empty();

            for node in partition.iter() {
                for (next_node, char) in nfa.get_edges(*node) {
                    if *char == Some(ch) {
                       res.insert(*next_node);
                    }
                }
            }
           res
        }

        /**
         * Follows the no-char paths from the nodes
         *   :param partition: The initial partition
         *   :return: The initial partition with their no-char successors added
         */
        fn get_partition_token<T: CustomTokenType>(nfa: &NFA<T>, partition: &Partition) -> Result<Option<T>, NfaToDfaError<T>> {
            let mut iter = partition.iter().filter_map(|x| nfa.get_node_token(*x));
            let token = iter.next();
            if let Some(second_token) = iter.next() {
                let first = token.unwrap();
                if first != second_token {
                    return Err(NfaToDfaError::StateConflict(first, second_token));
                    //panic!("Multiple tokens found for node partition: {:?} {:?}   !", first, second_token);
                }
            }
            Ok(token)
        }

        let alphabet = self.get_alphabet();

        let mut start_nodes = Partition::create_empty();
        start_nodes.insert(0);
        no_char_closure(self, &mut start_nodes);

        let mut out_nodes = HashMap::new();
        out_nodes.insert(start_nodes.clone(), 0);
        res.add_node(get_partition_token(self, &start_nodes)?);
        let mut next_id = 1;

        let mut work_list = vec![(start_nodes, 0)];

        while let Some((current, current_index)) = work_list.pop() {
            //println!("Step {:?} {}", current, current_index);
            for ch in &alphabet {
                let mut after_section = advance_partition(self, &current, *ch);
                no_char_closure(self, &mut after_section);

                if after_section.is_empty() {
                    continue
                }

                let after_index = out_nodes.get(&after_section)
                    .map(|x| Ok(*x))
                    .unwrap_or_else( || {
                        res.add_node(get_partition_token(self, &after_section)?);
                        out_nodes.insert(after_section.clone(), next_id);
                        next_id += 1;
                        work_list.push((after_section, next_id - 1));
                        Ok(next_id - 1)
                })?;

                res.add_edge(current_index, after_index, Some(*ch))
            }
        }

        // This should never fail if the conversion has been done correctly
        Ok(res.to_dfa_direct().expect("Conversion failed"))
    }

    pub fn from_text(text: &[u8], token: Option<T>) -> NFA<T> {
        let mut nfa = NFA::new();
        nfa.add_empty_nodes(text.len() + 1);
        nfa.set_node_token(text.len() as u32, token);

        for (i, x) in text.iter().enumerate() {
            let i = i as u32;
            nfa.add_edge(i, i + 1, Some(*x));
        }

        nfa
    }
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum NfaToDfaError<T: CustomTokenType> {
    StateConflict(T, T)
}

impl<T: CustomTokenType> Display for NonDeterministicFiniteAutomaton<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.write_fmt(format_args!("Nodes: {}\n", self.nodes.len()))?;
        for (index, token) in self.nodes.iter().enumerate() {
            if let Some(t) = token {
                f.write_fmt(format_args!("{} = {:?}\n", index, t))?;
            }
        }
        for (from, neighs) in self.neighbours.iter().enumerate() {
            for (to, ch) in neighs.iter() {
                if let Some(ch) = ch {
                    f.write_fmt(format_args!("{} -> {} ({:?})\n", from, to, *ch as char))?;
                } else {
                    f.write_fmt(format_args!("{} -> {}\n", from, to))?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::scanner::{Token, TokenType, NFA};

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    enum TestTokenType {
        NOT, NEW
    }

    #[test]
    fn basic_test() {
        // not|new
        //   -n-> 1 -o-> 2 -t-> 3
        // 0 |
        //   -n-> 4 -e-> 5 -w-> 6
        let mut x = NFA::new();
        let n0 = x.add_node(None);
        let n1 = x.add_node(None);
        let n2 = x.add_node(None);
        let n3 = x.add_node(Some(TestTokenType::NOT));
        let n4 = x.add_node(None);
        let n5 = x.add_node(None);
        let n6 = x.add_node(Some(TestTokenType::NEW));
        x.add_edge(n0, n1, Some('n' as u8));
        x.add_edge(n1, n2, Some('o' as u8));
        x.add_edge(n2, n3, Some('t' as u8));
        x.add_edge(n0, n4, Some('n' as u8));
        x.add_edge(n4, n5, Some('e' as u8));
        x.add_edge(n5, n6, Some('w' as u8));
        let res = x.to_dfa().unwrap();

        assert_eq!(res.get_node_count(), 6);
        assert_eq!(res.tokenize("not", 0), [
            Token { text: "not".to_owned(), ttype: TokenType::Custom(TestTokenType::NOT) },
            Token { text: "".to_owned(), ttype: TokenType::End }
        ]);

        assert_eq!(res.tokenize("new", 0), [
            Token { text: "new".to_owned(), ttype: TokenType::Custom(TestTokenType::NEW) },
            Token { text: "".to_owned(), ttype: TokenType::End }
        ]);
    }
}

