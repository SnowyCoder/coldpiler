use crate::parser::grammar::{GrammarToken, GrammarTokenType};
use crate::parser::tree::{NodeIndex, SyntaxTree};
use crate::scanner::{CustomTokenType, Token, TokenType};

use super::grammar::Grammar;

// Top-down parsing using a stupid method

struct ParseData<'a, T, N> where T : CustomTokenType, N : GrammarTokenType {
    grammar: &'a Grammar<T, N>,
    tokens: &'a [Token<T>],
    next_token_index: usize,

    tree: SyntaxTree<T, N>,
    focus: Option<(GrammarToken<T, N>, NodeIndex)>,

    node_stack: Vec<(GrammarToken<T, N>, NodeIndex)>,
    current_token: &'a Token<T>,
}


impl<'a, T: CustomTokenType, N: GrammarTokenType> ParseData<'a, T, N> {
    // This method advances the focus and focus_node, consuming the current one if
    // requested.
    fn advance_token(&mut self, consume_last: bool) {
        if consume_last && self.tokens.len() > self.next_token_index + 1 {
            self.next_token_index += 1;
        }
        self.current_token = &self.tokens[self.next_token_index];
        if self.current_token.ttype == TokenType::Error {
            eprintln!("Error: unrecognized token {}", self.current_token.text);
            self.advance_token(true)
        } else if self.current_token.ttype == TokenType::End {
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
                    if self.current_token.ttype != TokenType::Custom(token) {
                        // TODO: better error management
                        panic!(format!("Error: expected {:?} but found {:?}", token, self.current_token))
                    } else {
                        self.tree.mut_node(focus_node).text = Some(self.current_token.text.clone());
                        self.consume_focus();
                        self.advance_token(true);
                    }
                },
                GrammarToken::NonTerminal(focus) => {
                    // Check what comes next of this token, if something is found, create
                    // enough nodes for them (as they all will be children of the focus node)
                    // then reverse the nodes and put them in the node_stack.
                    // TODO: better error checking

                    // TODO: better type usage? we shouldn't be rechecking the type as it is already
                    //       being checked in the advance_token routine.
                    let current_token = match self.current_token.ttype {
                        TokenType::Custom(t) => t,
                        _ => unreachable!(),
                    };
                    // Note: using "self" in the lambda captures it in the closure so it will be captured
                    // as mutable, to avoid this (as we need to use node_stack also as mutable) we
                    // capture in the closure only "tree" borrowing a reference to it.
                    let tree = &mut self.tree;

                    let mut nodes: Vec<(GrammarToken<T, N>, NodeIndex)> =
                        self.grammar.find_next(focus.index(), current_token)
                        .unwrap_or_else(|| panic!(format!("Unexpected token {:?}", current_token)))
                        .iter()
                        .map(|&x| (x, tree.create_node(x, None, Some(focus_node))))
                        .collect();

                    self.node_stack.extend(nodes.drain(..).rev());
                    self.consume_focus()
                }
            }
        }
    }
}


impl<T: CustomTokenType, N: GrammarTokenType> Grammar<T, N> {
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
        let focus_node = tree.create_node(focus, None, None);

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

        Some(data.tree)
    }
}


#[cfg(test)]
mod tests {
    use std::slice::Iter;

    use TestTokenType::{NUMBER, PLUS, SPACE};

    use crate::parser::grammar::{Enumerable, Grammar, GrammarToken};
    use crate::parser::grammar::GrammarToken::{NonTerminal, Terminal};
    use crate::parser::tree::SyntaxNode;
    use crate::scanner::{Token, TokenType};
    use std::iter::Cloned;

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    enum TestTokenType {
        SPACE,
        NUMBER,
        PLUS,
    }

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    enum TestGrammarTokenType {
        Statement,
        StatementTail,
        Space,
    }

    type TGTT = TestGrammarTokenType;

    impl Enumerable for TestGrammarTokenType {
        type Iterator = Cloned<Iter<'static, Self>>;

        fn index(&self) -> usize {
            return match self {
                TGTT::Statement => 0,
                TGTT::StatementTail => 1,
                TGTT::Space => 2,
            }
        }

        fn enumerate() -> Self::Iterator {
            static TYPES: [TGTT; 3] = [TGTT::Statement, TGTT::StatementTail, TGTT::Space];
            TYPES.iter().cloned()
        }
    }


    #[test]
    fn basic_test() {
        let grammar = Grammar::from_raw(NonTerminal(TestGrammarTokenType::Statement), vec![
            vec![vec![GrammarToken::Terminal(NUMBER), GrammarToken::NonTerminal(TGTT::StatementTail)]],
            vec![vec![GrammarToken::Terminal(PLUS), GrammarToken::Terminal(NUMBER), GrammarToken::NonTerminal(TestGrammarTokenType::StatementTail)],
                 vec![]],
            vec![vec![GrammarToken::Terminal(SPACE)]]
        ]);

        let tokens = vec![
            Token { text: "40".to_string(), ttype: TokenType::Custom(NUMBER) },
            Token { text: "+".to_string(), ttype: TokenType::Custom(PLUS) },
            Token { text: "3".to_string(), ttype: TokenType::Custom(NUMBER) },
        ];

        let ast = grammar.run_top_down(&tokens).unwrap();
        assert_eq!(&SyntaxNode {
            parent: None,
            gtype: NonTerminal(TGTT::Statement),
            text: None,
            children: vec![1, 2]
        }, ast.node(0));
        assert_eq!(&SyntaxNode {
            parent: Some(0),
            gtype: Terminal(NUMBER),
            text: Some("40".to_string()),
            children: vec![]
        }, ast.node(1));
        assert_eq!(&SyntaxNode {
            parent: Some(0),
            gtype: NonTerminal(TGTT::StatementTail),
            text: None,
            children: vec![3, 4, 5]
        }, ast.node(2));
        assert_eq!(&SyntaxNode {
            parent: Some(2),
            gtype: Terminal(PLUS),
            text: Some("+".to_string()),
            children: vec![]
        }, ast.node(3));
        assert_eq!(&SyntaxNode {
            parent: Some(2),
            gtype: Terminal(NUMBER),
            text: Some("3".to_string()),
            children: vec![]
        }, ast.node(4));
        assert_eq!(&SyntaxNode {
            parent: Some(2),
            gtype: NonTerminal(TGTT::StatementTail),
            text: None,
            children: vec![]
        }, ast.node(5));

        let tokens = vec![
            Token { text: "40".to_string(), ttype: TokenType::Custom(NUMBER) },
            Token { text: "+".to_string(), ttype: TokenType::Custom(PLUS) },
            Token { text: "1".to_string(), ttype: TokenType::Custom(NUMBER) },
            Token { text: "+".to_string(), ttype: TokenType::Custom(PLUS) },
            Token { text: "1".to_string(), ttype: TokenType::Custom(NUMBER) },
        ];

        let ast = grammar.run_top_down(&tokens).unwrap();
        assert_eq!(9, ast.node_count());
    }
}
