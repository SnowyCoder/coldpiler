use coldpiler_parser::parser;

use crate::lang::*;

use super::tree::*;
use crate::context::Context;

pub type T = ScannerTokenType;
pub type NT = ParserTokenType;
pub type SyntaxTree = parser::SyntaxTree<T, NT>;
pub type SyntaxNode = parser::SyntaxNode<T, NT>;
pub type GrammarToken = parser::GrammarToken<T, NT>;

#[derive(Copy, Clone)]
struct Ctx<'a> {
    context: &'a Context,
    tree: &'a SyntaxTree,
}

impl Ctx<'_> {
    fn get_text(&self, node: &SyntaxNode) -> String {
        String::from_utf8(self.context.trie.find_key(node.text.expect("Text expected").trie_index)).expect("Invalid source code")
    }
}

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

fn build_lit(ctx: Ctx, node: &SyntaxNode) -> Value {
    let child = ctx.tree.node(node.children[0]);
    match child.gtype {
        GrammarToken::Terminal(T::NumberLiteral) => {
            Value::I32(parse_number_literal(&ctx.get_text(child)))
        }
        GrammarToken::Terminal(T::BoolLiteral) => {
            Value::Bool(parse_bool_literal(&ctx.get_text(child)))
        }
        _ => unreachable!(),
    }
}

fn build_if(ctx: Ctx, node: &SyntaxNode) -> IfExpr {
    let expr = build_expr(ctx, ctx.tree.node(node.children[1]));
    let block = build_block(ctx, ctx.tree.node(node.children[2]));
    let tail = ctx.tree.node(node.children[3])
        .children.get(1)
        .map(|else_block_index| {
            build_block(ctx, ctx.tree.node(*else_block_index))
        });

    IfExpr {
        blocks: vec![
            IfBlock { cond: expr, then: block }
        ],
        tail
    }
}

fn build_expr_base(ctx: Ctx, node: &SyntaxNode) -> Expr {
    let first_child = ctx.tree.node(node.children[0]);
    let detail = match first_child.gtype {
        GrammarToken::NonTerminal(NT::Block) => ExprDetail::Block(build_block(ctx, first_child)),
        GrammarToken::NonTerminal(NT::Lit) => ExprDetail::Lit(build_lit(ctx, first_child)),
        GrammarToken::NonTerminal(NT::IfExpr) => ExprDetail::If(build_if(ctx, first_child)),
        GrammarToken::NonTerminal(NT::PrintExpr) => ExprDetail::Print(Box::new(build_expr(ctx, ctx.tree.node(first_child.children[2])))),
        GrammarToken::Terminal(T::Identifier) =>  ExprDetail::Ident(build_ident(ctx, first_child)),
        _ => unreachable!(),
    };
    Expr(node.span, detail)
}

fn build_expr_op(ctx: Ctx, node: &SyntaxNode) -> Expr {
    let first_child = ctx.tree.node(node.children[0]);
    match first_child.gtype {
        GrammarToken::NonTerminal(NT::ExprOp) => {
            let lhs = build_expr_op(ctx, first_child);
            let op = build_ident(ctx, ctx.tree.node(node.children[1]));
            let rhs = build_expr_base(ctx, ctx.tree.node(node.children[2]));
            Expr(node.span, ExprDetail::Operation(Box::new(lhs), op, Box::new(rhs)))
        },
        GrammarToken::NonTerminal(NT::ExprBase) => {
            build_expr_base(ctx, first_child)
        },
        _ => unreachable!(),
    }
}

fn build_expr(ctx: Ctx, node: &SyntaxNode) -> Expr {
    let first_child = ctx.tree.node(node.children[0]);
    match first_child.gtype {
        GrammarToken::Terminal(T::Identifier) => {
            // Assignment: Id Eq <Expr>
            Expr(node.span, ExprDetail::Assign(Box::new(Assign {
                name: build_ident(ctx, first_child),
                expr: build_expr(ctx, ctx.tree.node(node.children[2])),
            })))
        },
        GrammarToken::NonTerminal(NT::ExprOp) => build_expr_op(ctx, first_child),
        _ => unreachable!(),
    }
}

fn build_ident(_ctx: Ctx, node: &SyntaxNode) -> Identifier {
    Identifier(node.text.unwrap())
}

fn build_decl(ctx: Ctx, node: &SyntaxNode) -> Declaration {
    Declaration {
        mutable: true,
        assign: Assign {
            name: build_ident(ctx, ctx.tree.node(node.children[1])),
            expr: build_expr(ctx, ctx.tree.node(node.children[3]))
        }
    }
}

fn build_block(ctx: Ctx, node: &SyntaxNode) -> Block {
    let mut current_entry = ctx.tree.node(node.children[1]);
    let mut entries = Vec::new();
    loop {
        let expr_or_decl = ctx.tree.node(current_entry.children[0]);

        let child = expr_or_decl.children.get(0)
            .map(|x| ctx.tree.node(*x));
        let entry = match child.map(|x| x.gtype) {
            None => BlockEntry::Unit,
            Some(GrammarToken::NonTerminal(NT::Expr)) => BlockEntry::Expr(build_expr(ctx, child.unwrap())),
            Some(GrammarToken::NonTerminal(NT::Declaration)) => BlockEntry::Decl(build_decl(ctx, child.unwrap())),
            _ => unreachable!(),
        };
        entries.push(entry);
        if let Some(next_node_id) = current_entry.children.get(2) {
            current_entry = ctx.tree.node(*next_node_id);
        } else {
            break;
        }
    }
    Block {
        exprs: entries
    }
}

pub fn build_main(context: &Context, tree: &SyntaxTree) -> Block {
    let root = tree.find_root().expect("Cannot find root node");
    let root = tree.node(root);
    assert_eq!(root.gtype, GrammarToken::NonTerminal(NT::Program));
    let block = tree.node(root.children[2]);
    let ctx = Ctx {
        context,
        tree,
    };
    build_block(ctx, block)
}