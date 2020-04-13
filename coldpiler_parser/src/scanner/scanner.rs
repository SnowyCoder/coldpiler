use std::fmt::Debug;

use coldpiler_util::radix_tree::RadixTree;

use crate::loc::SpanLoc;
use crate::scanner::NFA;

pub trait ScannerTokenType: Copy + Debug + Eq + Ord {
}

impl<T: Copy + Debug + Eq + Ord> ScannerTokenType for T {}

#[derive(Clone, Debug)]
pub struct Scanner<T: ScannerTokenType> {
    node_count: u32,
    tokens: Vec<Option<T>>,
    node_map: Vec<Option<u32>>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TokenLoc {
    pub trie_index: u32,
    pub span: SpanLoc,
}

impl TokenLoc {
    pub fn of(trie_index: u32, span_start_line: u32, span_start_col: u32, span_end_line: u32, span_end_col: u32) -> Self {
        TokenLoc {
            trie_index,
            span: SpanLoc::of(span_start_line, span_start_col, span_end_line, span_end_col),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Token<T: ScannerTokenType> {
    pub text: TokenLoc,
    pub ttype: T,
}

impl<T: ScannerTokenType> Scanner<T> {
    pub fn new(node_count: u32) -> Scanner<T> {
        Scanner {
            node_count,
            tokens: vec![None; node_count as usize],
            node_map: vec![None; (node_count * 256) as usize]
        }
    }

    pub fn from_raw(tokens: Vec<Option<T>>, node_map: Vec<Option<u32>>) -> Scanner<T> {
        assert_eq!(tokens.len() * 256, node_map.len());
        Scanner {
            node_count: tokens.len() as u32,
            tokens, node_map,
        }
    }

    pub fn into_raw(self) -> (Vec<Option<T>>, Vec<Option<u32>>) {
        (self.tokens, self.node_map)
    }

    pub fn get_node_count(&self) -> u32 {
        self.node_count
    }

    pub fn set_token(&mut self, node: u32, token: T) {
        self.tokens[node as usize] = Some(token)
    }

    pub fn add_edge(&mut self, from: u32, to: u32, ch: u8) {
        let new_to = self.add_edge_try(from, to, ch);
        if new_to != to {
            panic!(format!("conflict! node {}, {} is already connected to {}, trying to reconnect to {}", from, ch as char, new_to, to))
        }
    }

    pub fn add_edge_try(&mut self, from: u32, to: u32, ch: u8) -> u32 {
        let ind = (from * 256 + ch as u32) as usize;
        if let Some(old_to) = self.node_map[ind] {
            if old_to != to {
                return old_to;// Well, conflict
            }
        }
        self.node_map[ind] = Some(to);
        to
    }

    pub fn nodes(&self) -> std::slice::Iter<Option<T>> {
        self.tokens.iter()
    }

    pub fn get_next(&self, state: u32, ch: u8) -> Option<u32> {
        self.node_map[(state * 256 + ch as u32) as usize]
    }

    pub fn get_token(&self, state: u32) -> Option<T> {
        self.tokens[state as usize]
    }

    pub fn to_nfa(&self) -> NFA<T> {
        let mut nfa = NFA::new();
        nfa.reserve_nodes(self.node_count as usize);

        for token in self.tokens.iter() {
            nfa.add_node(token.clone());
        }

        for state in 0..self.node_count {
            for ch in 0..255u8 {
                if let Some(next_id) = self.get_next(state, ch) {
                    nfa.add_edge(state, next_id, Some(ch));
                }
            }
        }

        nfa
    }

    pub fn run_single_token(&self, data: &str, init_state: u32) -> Option<T> {
        let mut state = Some(init_state);
        for ch in data.bytes() {
            state = state.and_then(|x| self.get_next(x, ch))
        }
        state.and_then(|x| self.get_token(x))
    }

    pub fn tokenize(&self, trie: &mut RadixTree<u8>, data: &str, init_state: u32) -> (Vec<Token<T>>, Vec<TokenLoc>) {
        let bytes = data.as_bytes();
        let mut tokens: Vec<Token<T>> = Vec::new();
        let mut unrecognized: Vec<TokenLoc> = Vec::new();

        let mut token_start_index = 0;
        let mut token_start_line = 0;
        let mut token_start_column = 0;
        let mut last_successful_state: Option<u32> = None;
        let mut last_successful_index = 0;
        let mut last_successful_line = 0;
        let mut last_successful_column = 0;

        let mut index = 0;
        let mut line = 0;
        let mut column = 0;

        let mut state = init_state;
        while index < bytes.len() {
            // eprintln!("Index: {}->{} {}, {}, last: {} {:?}", token_start_index, index, state, bytes[index], last_successful_index, last_successful_state);
            if let Some(new_state) = self.get_next(state, bytes[index]) {
                state = new_state;
                if self.get_token(state).is_some() {
                    last_successful_index = index;
                    last_successful_line = line;
                    last_successful_column = column;
                    last_successful_state = Some(state);
                }

                if index + 1 < bytes.len() {
                    // TODO: properly manage multiple byte chars
                    // TODO: remove code duplitaction
                    if bytes[index] == b'\n' {
                        line += 1;
                        column = 0;
                    } else {
                        column += 1;
                    }
                }
            } else {
                // Error, try to backtrack
                if let Some(backtracked_state) = last_successful_state {
                    index = last_successful_index;
                    let trie_index = trie.insert(&bytes[token_start_index..=index]);

                    tokens.push(Token {
                        text: TokenLoc {
                            trie_index,
                            span: SpanLoc {
                                start_line: token_start_line,
                                start_column: token_start_column,
                                end_line: last_successful_line,
                                end_column: last_successful_column,
                            }
                        },
                        ttype: self.get_token(backtracked_state).unwrap(),
                    });
                    state = init_state;
                    last_successful_state = None;
                    last_successful_index = 0;
                    token_start_index = index + 1;

                    line = last_successful_line;
                    column = last_successful_column;
                    if bytes[index] == b'\n' {
                        line += 1;
                        column = 0;
                    } else {
                        column += 1;
                    }

                    token_start_line = line;
                    token_start_column = column;
                } else {
                    let trie_index = trie.insert(&bytes[token_start_index..=index]);
                    unrecognized.push(TokenLoc {
                        trie_index,
                        span: SpanLoc {
                            start_line: token_start_line,
                            start_column: token_start_column,
                            end_line: line,
                            end_column: column,
                        }
                    });
                    token_start_index = index + 1;
                    if bytes[index] == b'\n' {
                        line += 1;
                        column = 0;
                    } else {
                        column += 1;
                    }
                    token_start_line = line;
                    token_start_column = column;
                    state = init_state;
                }
            }

            index += 1;
        }

        if token_start_index != bytes.len() {
            let trie_index = trie.insert(&data[token_start_index..index]);
            let text = TokenLoc {
                trie_index,
                span: SpanLoc {
                    start_line: token_start_line,
                    start_column: token_start_column,
                    end_line: line,
                    end_column: column,
                },
            };

            if let Some(last_succ) = last_successful_state {
                if last_successful_index == index - 1 {
                    tokens.push(Token {
                        text,
                        ttype: self.get_token(last_succ).unwrap()
                    });
                } else {
                    unrecognized.push(text);
                }
            } else {
                unrecognized.push(text);
            }
        }

        (tokens, unrecognized)
    }
}

#[cfg(test)]
mod tests {
    use coldpiler_util::radix_tree::RadixTree;

    use crate::scanner::{Scanner, Token};
    use crate::scanner::scanner::TokenLoc;

    #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
    enum TestTokenType {
        AB, ABABA
    }

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn basic_test() {
        let mut x = Scanner::new(3);
        x.add_edge(0, 1, b'a');
        x.add_edge(1, 2, b'b');
        x.set_token(2, TestTokenType::AB);

        let mut trie = RadixTree::new();
        let (res, err) = x.tokenize(&mut trie, "aba", 0);
        assert_eq!(res, [
            Token { text: TokenLoc::of(trie.get("ab").unwrap(), 0, 0, 0, 1), ttype: TestTokenType::AB },
        ]);
        assert_eq!(err, [
            TokenLoc::of(trie.get("a").unwrap(), 0, 2, 0, 2)
        ])
    }

    #[test]
    fn test_2() {
        let mut x = Scanner::new(6);
        x.add_edge(0, 1, b'a');
        x.add_edge(1, 2, b'b');
        x.add_edge(2, 3, b'a');
        x.add_edge(3, 4, b'b');
        x.add_edge(4, 5, b'a');
        x.set_token(2, TestTokenType::AB);
        x.set_token(5, TestTokenType::ABABA);

        let mut trie = RadixTree::new();
        let (res, err) = x.tokenize(&mut trie, "abadababa", 0);// ababa odoru akachan ningen
        assert_eq!(res, [
            Token { text: TokenLoc::of(trie.get("ab").unwrap(), 0, 0, 0, 1), ttype: TestTokenType::AB },
            // Error: ad
            Token { text: TokenLoc::of(trie.get("ababa").unwrap(), 0, 4, 0, 8), ttype: TestTokenType::ABABA }
        ]);
        assert_eq!(err, [
            TokenLoc::of(trie.get("ad").unwrap(), 0, 2, 0, 3)
        ])
    }

    #[test]
    fn test_3() {
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
        enum TestTokenType {
            NOT, NEW
        }

        // not|new
        //          -o-> 2 -t-> 3|
        // 0 -n-> 1 |
        //          -e-> 4 -w-> 5|
        let mut x = Scanner::new(6);
        x.add_edge(0, 1, b'n');
        x.add_edge(1, 2, b'o');
        x.add_edge(2, 3, b't');
        x.add_edge(1, 4, b'e');
        x.add_edge(4, 5, b'w');
        x.set_token(3, TestTokenType::NOT);
        x.set_token(5, TestTokenType::NEW);

        let mut trie = RadixTree::new();

        assert_eq!(x.tokenize(&mut trie, "not", 0), (vec![
            Token { text: TokenLoc::of(trie.get("not").unwrap(), 0, 0, 0, 2), ttype: TestTokenType::NOT }
        ], vec![]));

        assert_eq!(x.tokenize(&mut trie, "new", 0), (vec![
            Token { text: TokenLoc::of(trie.get("new").unwrap(), 0, 0, 0, 2), ttype: TestTokenType::NEW }
        ], vec![]));

        assert_eq!(x.tokenize(&mut trie, "now", 0), (vec![], vec![
            TokenLoc::of(trie.get("now").unwrap(), 0, 0, 0, 2)
        ]));

        assert_eq!(x.tokenize(&mut trie, "no", 0), (vec![], vec![
            TokenLoc::of(trie.get("no").unwrap(), 0, 0, 0, 1)
        ]));
    }
}

