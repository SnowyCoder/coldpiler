use crate::loc::SpanLoc;
use crate::parser::grammar::{GrammarToken, GrammarTokenType};
use crate::parser::tree::{NodeIndex, SyntaxTree};
use crate::scanner::{ScannerTokenType, Token};

use super::grammar::Grammar;

// Top-down parsing using a stupid method

struct ParseData<'a, T, N> where T : ScannerTokenType, N : GrammarTokenType {
    grammar: &'a Grammar<T, N>,
    tokens: &'a [Token<T>],
    next_token_index: usize,

    tree: SyntaxTree<T, N>,
    focus: Option<(GrammarToken<T, N>, NodeIndex)>,

    node_stack: Vec<(GrammarToken<T, N>, NodeIndex)>,
    current_token: &'a Token<T>,
}


impl<'a, T: ScannerTokenType, N: GrammarTokenType> ParseData<'a, T, N> {
    // This method advances the focus and focus_node, consuming the current one if
    // requested.
    fn advance_token(&mut self, consume_last: bool) {
        if consume_last && self.tokens.len() > self.next_token_index + 1 {
            self.next_token_index += 1;
        }
        self.current_token = match self.tokens.get(self.next_token_index) {
            None => {
                // Try to short-circuit every token matching it to 0
                // crash if it's not possible
                while let Some((focus, _focus_node)) = self.focus {
                    if self.grammar.can_be_zero(focus) {
                        self.consume_focus()
                    } else {
                        // Todo: better error management.
                        panic!(format!("Token stream terminates too early, expected {:?}", focus));
                    }
                }
                return
            },
            Some(x) => x,
        };
        if self.grammar.ignored.contains(&self.current_token.ttype) {
            // Ignore token: a.k.a. readvance consuming the last token (this one)
            self.advance_token(true);
        }
    }

    fn consume_focus(&mut self) {
        self.focus = self.node_stack.pop();
    }


    fn run(&mut self) {
        while let Some((focus, focus_node)) = self.focus {
            // manage white-spaces (?)
            //eprintln!("Focus: {:?}, {}", focus, focus_node);

            match focus {
                GrammarToken::Terminal(token) => {
                    if self.current_token.ttype != token {
                        // TODO: better error management
                        panic!(format!("Error: expected {:?} but found {:?}", token, self.current_token))
                    } else {
                        let node = self.tree.mut_node(focus_node);
                        node.text = Some(self.current_token.text);
                        node.span = self.current_token.text.span;
                        self.consume_focus();
                        self.advance_token(true);
                    }
                },
                GrammarToken::NonTerminal(focus) => {
                    // Check what comes next of this token, if something is found, create
                    // enough nodes for them (as they all will be children of the focus node)
                    // then reverse the nodes and put them in the node_stack.
                    // TODO: better error checking
                    let current_token = self.current_token.ttype;
                    // Note: using "self" in the lambda captures it in the closure so it will be captured
                    // as mutable, to avoid this (as we need to use node_stack also as mutable) we
                    // capture in the closure only "tree" borrowing a reference to it.
                    let tree = &mut self.tree;

                    let mut nodes: Vec<(GrammarToken<T, N>, NodeIndex)> =
                        self.grammar.find_next(focus.index(), current_token)
                        .unwrap_or_else(|| panic!(format!("Unexpected token {:?}", current_token)))
                        .iter()
                        .map(|&x| (x, tree.create_node(x, SpanLoc::zero(), None, Some(focus_node))))
                        .collect();

                    self.node_stack.extend(nodes.drain(..).rev());
                    self.consume_focus()
                }
            }
        }
    }
}


impl<T: ScannerTokenType, N: GrammarTokenType> Grammar<T, N> {
    // Wow, how does this amazing advanced algorithm work?
    // It just traverses the grammar top-down creating nodes when it needs them, literally the first
    // method that came to mind while reading.
    // It keeps a focus node so that it knows the current node and a stack of node-tokens that will
    // need to be filled. Oh and it works only with LR(1) grammars.
    // It still doesn't implement any error reporting (yep, it's panic-time) so please never use
    // this for any practical code and prefer using lalr_table' method.
    pub fn run_top_down(&self, tokens: &[Token<T>]) -> Option<SyntaxTree<T, N>> {
        let mut tree = SyntaxTree::new();
        let focus = self.root;
        let focus_node = tree.create_node(focus, SpanLoc::zero(), None, None);

        let first_token = match tokens.get(0) {
            Some(x) => x,
            None => return None,
        };

        let mut data = ParseData {
            grammar: &self,
            tokens,
            next_token_index: 0,
            tree,
            focus: Some((focus, focus_node)),
            node_stack: vec![],
            current_token: &first_token,// Overridden, or better: checked in the first advance_token
        };

        data.advance_token(false);
        data.run();

        data.tree.recompute_all_spans(0);

        Some(data.tree)
    }
}


#[cfg(test)]
mod tests {
    use std::iter::Cloned;
    use std::slice::Iter;

    use coldpiler_util::Enumerable;

    use TestTokenType::{NUMBER, PLUS, SPACE};

    use crate::loc::SpanLoc;
    use crate::parser::grammar::{Grammar, GrammarToken};
    use crate::parser::grammar::GrammarToken::{NonTerminal, Terminal};
    use crate::parser::tree::SyntaxNode;
    use crate::scanner::{Token, TokenLoc};

    #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum TestTokenType {
        SPACE,
        NUMBER,
        PLUS,
    }

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    enum TestGrammarTokenType {
        Statement,
        StatementTail,
    }

    type TGTT = TestGrammarTokenType;

    impl Enumerable for TestGrammarTokenType {
        type Iterator = Cloned<Iter<'static, Self>>;

        fn index(&self) -> usize {
            match self {
                TGTT::Statement => 0,
                TGTT::StatementTail => 1,
            }
        }

        fn enumerate() -> Self::Iterator {
            static TYPES: [TGTT; 2] = [TGTT::Statement, TGTT::StatementTail];
            TYPES.iter().cloned()
        }
    }


    #[test]
    fn basic_test() {
        let grammar = Grammar::from_raw(NonTerminal(TestGrammarTokenType::Statement), vec![
            vec![vec![GrammarToken::Terminal(NUMBER), GrammarToken::NonTerminal(TGTT::StatementTail)]],
            vec![vec![GrammarToken::Terminal(PLUS), GrammarToken::Terminal(NUMBER), GrammarToken::NonTerminal(TestGrammarTokenType::StatementTail)],
                 vec![]]
        ], vec![SPACE]);

        // Original text: 40\n +3
        let tokens = vec![
            Token { text: TokenLoc::of(0, 0, 0, 0, 1), ttype: NUMBER },
            Token { text: TokenLoc::of(1, 0, 2, 1, 0), ttype: SPACE },
            Token { text: TokenLoc::of(2, 1, 1, 1, 1), ttype: PLUS },
            Token { text: TokenLoc::of(3, 1, 2, 1, 2), ttype: NUMBER },
        ];

        let ast = grammar.run_top_down(&tokens).unwrap();
        assert_eq!(&SyntaxNode {
            parent: None,
            gtype: NonTerminal(TGTT::Statement),
            text: None,
            span: SpanLoc::of(0, 0, 1, 2),
            children: vec![1, 2]
        }, ast.node(0));
        assert_eq!(&SyntaxNode {// 40
            parent: Some(0),
            gtype: Terminal(NUMBER),
            text: Some(TokenLoc::of(0, 0, 0, 0, 1)),
            span: SpanLoc::of(0, 0, 0, 1),
            children: vec![]
        }, ast.node(1));
        assert_eq!(&SyntaxNode {
            parent: Some(0),
            gtype: NonTerminal(TGTT::StatementTail),
            text: None,
            span: SpanLoc::of(1, 1, 1, 2),
            children: vec![3, 4, 5]
        }, ast.node(2));
        assert_eq!(&SyntaxNode {// +
            parent: Some(2),
            gtype: Terminal(PLUS),
            text: Some(TokenLoc::of(2, 1, 1, 1, 1)),
            span: SpanLoc::of(1, 1, 1, 1),
            children: vec![]
        }, ast.node(3));
        assert_eq!(&SyntaxNode {// 3
            parent: Some(2),
            gtype: Terminal(NUMBER),
            text: Some(TokenLoc::of(3, 1, 2, 1, 2)),
            span: SpanLoc::of(1, 2, 1, 2),
            children: vec![]
        }, ast.node(4));
        assert_eq!(&SyntaxNode {
            parent: Some(2),
            gtype: NonTerminal(TGTT::StatementTail),
            text: None,
            span: SpanLoc::of(1, 2, 1, 2),
            children: vec![]
        }, ast.node(5));

        // Original text: 40+1+1
        let tokens = vec![
            Token { text: TokenLoc::of(0, 0, 0, 0, 1), ttype: NUMBER },
            Token { text: TokenLoc::of(1, 0, 2, 0, 2), ttype: PLUS },
            Token { text: TokenLoc::of(2, 0, 3, 0, 3), ttype: NUMBER },
            Token { text: TokenLoc::of(1, 0, 4, 0, 4), ttype: PLUS },
            Token { text: TokenLoc::of(2, 0, 5, 0, 5), ttype: NUMBER },
        ];

        let ast = grammar.run_top_down(&tokens).unwrap();
        assert_eq!(9, ast.node_count());
    }
}
