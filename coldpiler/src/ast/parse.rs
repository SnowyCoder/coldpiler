use coldpiler_parser::parser;

use crate::lang::*;

use super::tree::*;

pub type T = ScannerTokenType;
pub type NT = ParserTokenType;
pub type SyntaxTree = parser::SyntaxTree<T, NT>;
pub type SyntaxNode = parser::SyntaxNode<T, NT>;
pub type GrammarToken = parser::GrammarToken<T, NT>;

fn parse_number_literal(text: &str) -> i32 {
    if text.starts_with("0b") {
        i32::from_str_radix(&text[2..], 2).unwrap()
    } else if text.starts_with("0x") {
        i32::from_str_radix(&text[2..], 16).unwrap()
    } else if text.starts_with("0o") {
        i32::from_str_radix(&text[2..], 8).unwrap()
    } else {
        text.parse().unwrap()
    }
}

fn parse_bool_literal(text: &str) -> bool {
    match text {
        "true" => true,
        "false" => false,
        _ => unreachable!(),
    }
}

fn build_lit(tree: &SyntaxTree, node: &SyntaxNode) -> Value {
    let child = tree.node(node.children[0]);
    match child.gtype {
        GrammarToken::Terminal(T::NumberLiteral) => {
            Value::I32(parse_number_literal(child.text.as_ref().unwrap()))
        }
        GrammarToken::Terminal(T::BoolLiteral) => {
            Value::Bool(parse_bool_literal(child.text.as_ref().unwrap()))
        }
        _ => unreachable!(),
    }
}

fn build_if(tree: &SyntaxTree, node: &SyntaxNode) -> IfExpr {
    let expr = build_expr(tree, tree.node(node.children[1]));
    let block = build_block(tree, tree.node(node.children[2]));
    let tail = tree.node(node.children[3])
        .children.get(1)
        .map(|else_block_index| {
            build_block(tree, tree.node(*else_block_index))
        });

    IfExpr {
        blocks: vec![
            IfBlock { cond: expr, then: block }
        ],
        tail
    }
}

fn build_expr_base(tree: &SyntaxTree, node: &SyntaxNode) -> Expr {
    let first_child = tree.node(node.children[0]);
    match first_child.gtype {
        GrammarToken::NonTerminal(NT::Block) => Expr::Block(build_block(tree, first_child)),
        GrammarToken::NonTerminal(NT::Lit) => Expr::Lit(build_lit(tree, first_child)),
        GrammarToken::NonTerminal(NT::IfExpr) => Expr::If(build_if(tree, first_child)),
        GrammarToken::NonTerminal(NT::PrintExpr) => Expr::Print(Box::new(build_expr(tree, tree.node(first_child.children[2])))),
        GrammarToken::Terminal(T::Identifier) =>  Expr::Ident(build_ident(first_child)),
        _ => unreachable!(),
    }
}

fn build_expr_op(tree: &SyntaxTree, node: &SyntaxNode) -> Expr {
    let first_child = tree.node(node.children[0]);
    match first_child.gtype {
        GrammarToken::NonTerminal(NT::ExprOp) => {
            let lhs = build_expr_op(tree, first_child);
            let op = build_ident(tree.node(node.children[1]));
            let rhs = build_expr_base(tree, tree.node(node.children[2]));
            Expr::Operation(Box::new(lhs), op, Box::new(rhs))
        },
        GrammarToken::NonTerminal(NT::ExprBase) => {
            build_expr_base(tree, first_child)
        },
        _ => unreachable!(),
    }
}

fn build_expr(tree: &SyntaxTree, node: &SyntaxNode) -> Expr {
    let first_child = tree.node(node.children[0]);
    match first_child.gtype {
        GrammarToken::Terminal(T::Identifier) => {
            // Assignment: Id Eq <Expr>
            Expr::Assign(Box::new(Assign {
                name: build_ident(first_child),
                expr: build_expr(tree, tree.node(node.children[2])),
            }))
        },
        GrammarToken::NonTerminal(NT::ExprOp) => build_expr_op(tree, first_child),
        _ => unreachable!(),
    }
}

fn build_ident(node: &SyntaxNode) -> Identifier {
    Identifier(node.text.clone().unwrap())
}

fn build_decl(tree: &SyntaxTree, node: &SyntaxNode) -> Declaration {
    Declaration {
        mutable: true,
        assign: Assign {
            name: build_ident(tree.node(node.children[1])),
            expr: build_expr(tree, tree.node(node.children[3]))
        }
    }
}

fn build_block(tree: &SyntaxTree, node: &SyntaxNode) -> Block {
    let mut current_entry = tree.node(node.children[1]);
    let mut entries = Vec::new();
    loop {
        let expr_or_decl = tree.node(current_entry.children[0]);

        let child = expr_or_decl.children.get(0)
            .map(|x| tree.node(*x));
        let entry = match child.map(|x| x.gtype) {
            None => BlockEntry::Unit,
            Some(GrammarToken::NonTerminal(NT::Expr)) => BlockEntry::Expr(build_expr(tree, child.unwrap())),
            Some(GrammarToken::NonTerminal(NT::Declaration)) => BlockEntry::Decl(build_decl(tree, child.unwrap())),
            _ => unreachable!(),
        };
        entries.push(entry);
        if let Some(next_node_id) = current_entry.children.get(2) {
            current_entry = tree.node(*next_node_id);
        } else {
            break;
        }
    }
    Block {
        exprs: entries
    }
}

pub fn build_main(tree: &SyntaxTree) -> Block {
    let root = tree.find_root().expect("Cannot find root node");
    let root = tree.node(root);
    assert_eq!(root.gtype, GrammarToken::NonTerminal(NT::Program));
    let block = tree.node(root.children[2]);
    build_block(tree, block)
}