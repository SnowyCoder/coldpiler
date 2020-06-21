use coldpiler_parser::parser;

use crate::ast::ExprId;
use crate::context::Context;
use crate::lang::*;

use super::tree::*;

pub type T = ScannerTokenType;
pub type NT = ParserTokenType;
pub type SyntaxTree = parser::SyntaxTree<T, NT>;
pub type SyntaxNode = parser::SyntaxNode<T, NT>;
pub type GrammarToken = parser::GrammarToken<T, NT>;

struct Ctx<'a> {
    context: &'a mut Context,
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

fn build_lit(ctx: &mut Ctx, node: &SyntaxNode) -> Value {
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

fn build_if(ctx: &mut Ctx, node: &SyntaxNode) -> IfExpr {
    let expr = build_expr(ctx, ctx.tree.node(node.children[1]));
    let block = build_block_expr(ctx, ctx.tree.node(node.children[2]));
    let tail = ctx.tree.node(node.children[3])
        .children.get(1)
        .map(|else_block_index| {
            build_block_expr(ctx, ctx.tree.node(*else_block_index))
        });

    IfExpr {
        blocks: vec![
            IfBlock { cond: expr, then: block }
        ],
        tail
    }
}

fn build_function_args(ctx: &mut Ctx, node: &SyntaxNode) -> Vec<ExprId> {
    let mut res = Vec::new();

    if node.children.len() < 3 {
        return res;
    }

    let mut curr = ctx.tree.node(node.children[1]);

    loop {
        res.push(build_expr(ctx, ctx.tree.node(curr.children[0])));
        if curr.children.len() < 3 {
            break;
        }
        curr = ctx.tree.node(curr.children[2]);
    }
    res
}

fn build_function_call(ctx: &mut Ctx, node: &SyntaxNode) -> FunctionCall {
    let name = build_ident(ctx, ctx.tree.node(node.children[0]));
    let args = build_function_args(ctx, ctx.tree.node(node.children[1]));
    FunctionCall {
        name,
        args,
        function_id: None
    }
}

fn build_expr_base(ctx: &mut Ctx, node: &SyntaxNode) -> ExprId {
    let first_child = ctx.tree.node(node.children[0]);
    let details = match first_child.gtype {
        GrammarToken::NonTerminal(NT::Block) => ExprDetail::Block(build_block(ctx, first_child)),
        GrammarToken::NonTerminal(NT::Lit) => ExprDetail::Lit(build_lit(ctx, first_child)),
        GrammarToken::NonTerminal(NT::IfExpr) => ExprDetail::If(build_if(ctx, first_child)),
        GrammarToken::NonTerminal(NT::FunctionCall) => ExprDetail::FunctionCall(build_function_call(ctx, first_child)),
        GrammarToken::Terminal(T::Identifier) =>  ExprDetail::Var(build_ident(ctx, first_child)),
        _ => unreachable!(),
    };
    ctx.context.ast.register_expr(Expr {
        loc: node.span,
        level: 0,
        details,
        res_type: None
    })
}

fn build_expr_op(ctx: &mut Ctx, node: &SyntaxNode) -> ExprId {
    let first_child = ctx.tree.node(node.children[0]);
    match first_child.gtype {
        GrammarToken::NonTerminal(NT::ExprOp) => {
            let lhs = build_expr_op(ctx, first_child);
            let op = build_ident(ctx, ctx.tree.node(node.children[1]));
            let rhs = build_expr_base(ctx, ctx.tree.node(node.children[2]));
            let fun = FunctionCall {
                name: op,
                args: vec![lhs, rhs],
                function_id: None
            };
            ctx.context.ast.register_expr(Expr {
                loc: node.span,
                level: 0,
                details: ExprDetail::FunctionCall(fun),
                res_type: None,
            })
        },
        GrammarToken::NonTerminal(NT::ExprBase) => {
            build_expr_base(ctx, first_child)
        },
        _ => unreachable!(),
    }
}

fn build_expr(ctx: &mut Ctx, node: &SyntaxNode) -> ExprId {
    let first_child = ctx.tree.node(node.children[0]);
    match first_child.gtype {
        GrammarToken::Terminal(T::Identifier) => {
            // Assignment: Id Eq <Expr>
            let expr = Expr {
                loc: node.span,
                level: 0,
                details: ExprDetail::Assign(Assign {
                    name: build_ident(ctx, first_child),
                    expr: build_expr(ctx, ctx.tree.node(node.children[2])),
                }),
                res_type: None,
            };
            ctx.context.ast.register_expr(expr)
        },
        GrammarToken::NonTerminal(NT::ExprOp) => build_expr_op(ctx, first_child),
        _ => unreachable!(),
    }
}

fn build_ident(_ctx: &mut Ctx, node: &SyntaxNode) -> Identifier {
    Identifier(node.text.unwrap())
}


fn build_type(ctx: &mut Ctx, node: &SyntaxNode) -> Type {
    let id = build_ident(ctx, node);
    let idx = id.0.trie_index;
    let bank = &ctx.context.bank;
    if idx == bank.unit {
        Type::Unit
    } else if idx == bank.i32 {
        Type::I32
    } else if idx == bank.bool {
        Type::Unit
    } else {
        Type::Custom(idx)
    }
}

fn build_decl(ctx: &mut Ctx, node: &SyntaxNode) -> Declaration {
    Declaration {
        mutable: true,
        assign: Assign {
            name: build_ident(ctx, ctx.tree.node(node.children[1])),
            expr: build_expr(ctx, ctx.tree.node(node.children[3]))
        }
    }
}

fn build_block(ctx: &mut Ctx, node: &SyntaxNode) -> Block {
    let mut current_entry = ctx.tree.node(node.children[1]);
    let mut entries = Vec::new();
    loop {
        let expr_or_decl = ctx.tree.node(current_entry.children[0]);

        let child = expr_or_decl.children.get(0)
            .map(|x| ctx.tree.node(*x));
        let entry = match child.map(|x| x.gtype) {
            None => ctx.context.ast.register_expr(Expr {
                loc: expr_or_decl.span,
                level: 0,
                details: ExprDetail::Lit(Value::Unit),
                res_type: Some(Type::Unit),
            }),
            Some(GrammarToken::NonTerminal(NT::Expr)) => build_expr(ctx, child.unwrap()),
            Some(GrammarToken::NonTerminal(NT::Declaration)) => {
                let expr = Expr {
                    loc: expr_or_decl.span,
                    level: 0,
                    details: ExprDetail::Decl(build_decl(ctx, child.unwrap())),
                    res_type: None
                };
                ctx.context.ast.register_expr(expr)
            },
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

fn build_block_expr(ctx: &mut Ctx, node: &SyntaxNode) -> ExprId {
    let expr = Expr {
        loc: node.span,
        level: 0,
        details: ExprDetail::Block(build_block(ctx, node)),
        res_type: None
    };
    ctx.context.ast.register_expr(expr)
}


fn build_function_decl_args(ctx: &mut Ctx, node: &SyntaxNode) -> Vec<(Identifier, Type)> {
    let mut res = Vec::new();

    if node.children.len() <= 2 {
        return res;
    }

    let mut entry = ctx.tree.node(node.children[1]);

    loop {
        let name = build_ident(ctx, ctx.tree.node(entry.children[0]));
        let arg_type = build_type(ctx, ctx.tree.node(entry.children[2]));
        res.push((name, arg_type));

        if entry.children.len() <= 3 {
            break;
        }
        entry = ctx.tree.node(entry.children[5]);
    }

    res
}

fn build_function_declaration(ctx: &mut Ctx, node: &SyntaxNode) -> FunctionDeclaration {
    let name = build_ident(ctx, ctx.tree.node(node.children[1]));
    let args = build_function_decl_args(ctx, ctx.tree.node(node.children[2]));
    let ret_node = ctx.tree.node(node.children[3]);

    let ret_type = if ret_node.children.is_empty() {
        Type::Unit
    } else {
        build_type(ctx, ctx.tree.node(ret_node.children[1]))
    };

    let body = build_block_expr(ctx, ctx.tree.node(node.children[4]));

    FunctionDeclaration {
        name, args,  body,
        ret_type: Some(ret_type),
        tac_id: 0
    }
}

pub fn build_file(context: &mut Context, tree: &SyntaxTree) {
    let root = tree.find_root().expect("Cannot find root node");
    let root = tree.node(root);
    assert_eq!(root.gtype, GrammarToken::NonTerminal(NT::Program));

    let mut current = root;

    let mut ctx = Ctx {
        context,
        tree,
    };

    while !current.children.is_empty() {
        let fun_decl = build_function_declaration(&mut ctx, tree.node(current.children[0]));
        ctx.context.ast.register_function(&ctx.context.bank, FunctionDefinition::Custom(fun_decl));
        current = tree.node(current.children[1]);
    }
}