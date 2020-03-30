use crate::scanner::TokenType::{Error, Custom};
use std::fmt::Debug;
use crate::scanner::NFA;

pub trait CustomTokenType : Copy + Debug + Eq + Ord {
}

impl<T: Copy + Debug + Eq + Ord> CustomTokenType for T {}

#[derive(Clone, Debug)]
pub struct Scanner<T: CustomTokenType> {
    node_count: u32,
    tokens: Vec<Option<T>>,
    node_map: Vec<Option<u32>>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TokenType<T: CustomTokenType> {
    Custom(T),
    Error,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Token<T: CustomTokenType> {
    pub text: String,
    pub ttype: TokenType<T>,
}

impl<T: CustomTokenType> Scanner<T> {
    pub fn new(node_count: u32) -> Scanner<T> {
        Scanner {
            node_count,
            tokens: vec![None; node_count as usize],
            node_map: vec![None; (node_count * 256) as usize]
        }
    }

    pub fn from_raw(tokens: Vec<Option<T>>, node_map: Vec<Option<u32>>) -> Scanner<T> {
        assert!(tokens.len() * 256 == node_map.len());
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

    pub fn tokenize(&self, data: &str, init_state: u32) -> Vec<Token<T>> {
        let bytes = data.as_bytes();
        let mut tokens: Vec<Token<T>> = Vec::new();

        let mut token_start_index = 0;
        let mut last_successful_state: Option<u32> = None;
        let mut last_successful_index = 0;

        let mut index = 0;
        let mut state = init_state;
        while index < bytes.len() {
            // eprintln!("Index: {}->{} {}, {}, last: {} {:?}", token_start_index, index, state, bytes[index], last_successful_index, last_successful_state);
            if let Some(new_state) = self.get_next(state, bytes[index]) {
                state = new_state;
                if self.get_token(state).is_some() {
                    last_successful_index = index;
                    last_successful_state = Some(state);
                }
            } else {
                // Error, try to backtrack
                if let Some(backtracked_state) = last_successful_state {
                    index = last_successful_index;
                    tokens.push(Token {
                        text: data[token_start_index..=index].to_string(),
                        ttype: Custom(self.get_token(backtracked_state).unwrap())
                    });
                    state = init_state;
                    last_successful_state = None;
                    last_successful_index = 0;
                    token_start_index = index + 1;
                } else {
                    tokens.push(Token {
                        text: data[token_start_index..=index].to_string(),
                        ttype: Error
                    });
                    token_start_index = index + 1;
                    state = 0;
                }
            }
            index += 1;
        }

        if token_start_index != bytes.len() {
            if last_successful_state.is_some() && last_successful_index == index - 1{
                tokens.push(Token {
                    text: data[token_start_index..index].to_string(),
                    ttype: Custom(self.get_token(last_successful_state.unwrap()).unwrap())
                });
            } else {
                tokens.push(Token {
                    text: data[token_start_index..index].to_string(),
                    ttype: Error
                });
            }
        }

        tokens
    }
}

#[cfg(test)]
mod tests {
    use crate::scanner::{Scanner, Token, TokenType};

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
        x.add_edge(0, 1, 'a' as u8);
        x.add_edge(1, 2, 'b' as u8);
        x.set_token(2, TestTokenType::AB);
        let res = x.tokenize("aba", 0);
        assert_eq!(res, [
            Token { text: "ab".to_owned(), ttype: TokenType::Custom(TestTokenType::AB) },
            Token { text: "a".to_owned(), ttype: TokenType::Error }
        ]);
    }

    #[test]
    fn test_2() {
        let mut x = Scanner::new(6);
        x.add_edge(0, 1, 'a' as u8);
        x.add_edge(1, 2, 'b' as u8);
        x.add_edge(2, 3, 'a' as u8);
        x.add_edge(3, 4, 'b' as u8);
        x.add_edge(4, 5, 'a' as u8);
        x.set_token(2, TestTokenType::AB);
        x.set_token(5, TestTokenType::ABABA);
        let res = x.tokenize("abadababa", 0);// ababa odoru akachan ningen
        assert_eq!(res, [
            Token { text: "ab".to_owned(), ttype: TokenType::Custom(TestTokenType::AB) },
            Token { text: "ad".to_owned(), ttype: TokenType::Error },
            Token { text: "ababa".to_owned(), ttype: TokenType::Custom(TestTokenType::ABABA) }
        ]);
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
        x.add_edge(0, 1, 'n' as u8);
        x.add_edge(1, 2, 'o' as u8);
        x.add_edge(2, 3, 't' as u8);
        x.add_edge(1, 4, 'e' as u8);
        x.add_edge(4, 5, 'w' as u8);
        x.set_token(3, TestTokenType::NOT);
        x.set_token(5, TestTokenType::NEW);

        assert_eq!(x.tokenize("not", 0), [
            Token { text: "not".to_owned(), ttype: TokenType::Custom(TestTokenType::NOT) }
        ]);

        assert_eq!(x.tokenize("new", 0), [
            Token { text: "new".to_owned(), ttype: TokenType::Custom(TestTokenType::NEW) }
        ]);

        assert_eq!(x.tokenize("now", 0), [
            Token { text: "now".to_owned(), ttype: TokenType::Error }
        ]);

        assert_eq!(x.tokenize("no", 0), [
            Token { text: "no".to_owned(), ttype: TokenType::Error }
        ]);
    }
}

