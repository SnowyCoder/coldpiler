pub use {
    grammar::Grammar, grammar::GrammarDefinition, grammar::GrammarRule,
    grammar::GrammarToken, grammar::GrammarTokenType,
    lalr_table::Action, lalr_table::LRConflict,
    lalr_table::ShiftReducer, tree::SyntaxNode, tree::SyntaxTree,
};

use crate::loc::SpanLoc;
use crate::scanner::ScannerTokenType;

mod tree;
mod grammar;
mod lalr_table;
mod top_down;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsingError<T: ScannerTokenType> {
    pub token: Option<T>,
    pub token_loc: SpanLoc,
    pub expected: Vec<Option<T>>,
}
