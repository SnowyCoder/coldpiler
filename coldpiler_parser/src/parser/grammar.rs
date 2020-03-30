use std::fmt::Debug;

use crate::scanner::CustomTokenType;

pub type TokenIndex = usize;


pub trait Enumerable : Sized + Copy {
    type Iterator: Iterator<Item = Self> + std::iter::ExactSizeIterator;

    fn index(&self) -> usize;

    fn enumerate() -> Self::Iterator;
}

pub trait GrammarTokenType : Copy + Debug + Eq + Enumerable {
}

impl<T: Copy + Debug + Eq + Enumerable> GrammarTokenType for T {}


#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GrammarToken<T : Copy + Debug + PartialEq, N : Copy + Debug + PartialEq> {
    Terminal(T),
    NonTerminal(N),
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash, Ord, PartialOrd)]
pub struct GrammarRuleIndex(pub u16, pub u16);

pub type GrammarRule<T, N> = Vec<GrammarToken<T, N>>;
pub type GrammarDefinition<T, N> = Vec<GrammarRule<T, N>>;


/// The syntax grammar used to build a Syntax Tree from a Token Stream
/// The grammar has various type definitions, each type definition has a set of rules that are
/// composed of various tokens (that might be either terminal or nonterminal).
/// Terminal tokens are to be matched with Syntax Tree's tokens while non-terminals match types in
/// the grammar, they might also allow recursion.
#[derive(Debug)]
pub struct Grammar<T: CustomTokenType, N: GrammarTokenType> {
    pub root: GrammarToken<T, N>,
    pub defs: Vec<GrammarDefinition<T, N>>,
    pub ignored: Vec<T>,
}

impl<T: CustomTokenType, N: GrammarTokenType> Grammar<T, N> {
    pub fn from_raw(root: GrammarToken<T, N>, defmap: Vec<GrammarDefinition<T, N>>, ignored: Vec<T>) -> Self {
        Grammar {
            root,
            defs: defmap,
            ignored,
        }
    }

    pub fn get_rule(&self, index: GrammarRuleIndex) -> &GrammarRule<T, N> {
        &self.defs[index.0 as usize][index.1 as usize]
    }

    /// Emulates the grammar and finds the next rule to be run from the state passed as input.
    /// Every non-empty rule takes precedence over empty rules, this method expects the grammar to
    /// be LR(1)
    pub fn find_next(&self, state: TokenIndex, expected_token: T) -> Option<&GrammarRule<T, N>> {
        let mut empty_rule: Option<&GrammarRule<T, N>> = None;

        for rule in &self.defs[state] {
            let mut rule_rejected = false;
            for token in rule {
                match token {
                    GrammarToken::Terminal(matched_token) => {
                        if *matched_token == expected_token {
                            return Some(&rule);
                        } else {
                            rule_rejected = true;
                            break;
                        }
                    }
                    GrammarToken::NonTerminal(next_state) => {
                        match self.find_next(next_state.index(), expected_token) {
                            None => {
                                rule_rejected = true;
                                break;
                            },
                            Some(x) => {
                                if !(x.is_empty()) {
                                    return Some(rule);
                                }
                            }
                        }
                    }
                }
            }
            if !rule_rejected && empty_rule.is_none() {
                empty_rule = Some(rule);
            }
        }

        empty_rule
    }

    /// Tries to match this token with no tokens (an empty construction)
    /// returns true if it can be done.
    pub fn can_be_zero_nonterminal(&self, index: TokenIndex) -> bool {
        for rule in &self.defs[index] {
            let mut rule_rejected = false;
            for token in rule {
                match token {
                    GrammarToken::Terminal(_matched_token) => {
                        rule_rejected = true;
                        break;
                    }
                    GrammarToken::NonTerminal(next_state) => {
                        if !self.can_be_zero_nonterminal(next_state.index()) {
                            rule_rejected = true;
                            break;
                        }
                    }
                }
            }
            if !rule_rejected {
                return true;
            }
        }
        false
    }

    pub fn can_be_zero(&self, token: GrammarToken<T, N>) -> bool {
        match token {
            GrammarToken::Terminal(_) => false,
            GrammarToken::NonTerminal(t) => self.can_be_zero_nonterminal(t.index()),
        }
    }
}

