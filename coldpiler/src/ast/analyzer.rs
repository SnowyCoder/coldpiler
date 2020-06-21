use coldpiler_parser::loc::SpanLoc;
use coldpiler_parser::scanner::TokenLoc;

use crate::ast::*;
use crate::error::{CompilationError, ErrorLoc};
use crate::ast::ast_data::{LevelId, AstData};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AnalyzeError {
    VariableNotFound(TokenLoc),
    FunctionNotFound(SpanLoc, String, Vec<Type>),
    MismatchedType(SpanLoc, Type, Type),
}

impl CompilationError for AnalyzeError {
    fn error_type(&self) -> String {
        "analyzer_error".to_string()
    }

    fn loc(&self) -> ErrorLoc {
        match self {
            AnalyzeError::VariableNotFound(x) => ErrorLoc::SingleLocation(x.span),
            AnalyzeError::FunctionNotFound(x, _, _) => ErrorLoc::SingleLocation(*x),
            AnalyzeError::MismatchedType(x, _, _) => ErrorLoc::SingleLocation(*x),
        }
    }

    fn summarize(&self) -> String {
        match self {
            AnalyzeError::VariableNotFound(_) => "Variable not found".to_string(),
            AnalyzeError::FunctionNotFound(_, _, _) => "Function not found".to_string(),
            AnalyzeError::MismatchedType(_, _, _) => "Mismatched types".to_string(),
        }
    }

    fn description(&self) -> String {
        match self {
            AnalyzeError::VariableNotFound(_) => "Variable not found".to_string(),
            AnalyzeError::FunctionNotFound(_, name, args) => format!("Function not found {}({:?})", name, args),
            AnalyzeError::MismatchedType(_, a, b) => format!("Expected '{}' but found '{}'", a, b),
        }
    }
}

pub struct Ctx<'a> {
    table: &'a mut AstData,
    errors: &'a mut Vec<AnalyzeError>,
    level: LevelId,
}

impl Ctx<'_> {
    pub fn with_new_level<R, F>(&mut self, func: F) -> R
        where F: FnOnce(&mut Self) -> R {

        let old_lev = self.level;
        self.level = self.table.create_level(Some(old_lev));
        let out = func(self);
        self.level = old_lev;
        out
    }
}

pub fn analyze_expr(ctx: &mut Ctx, expr_id: ExprId) -> Type {
    let expr = &mut ctx.table.exprs[expr_id];
    if let Some(x) = expr.res_type {
        return x;
    }
    expr.level = ctx.level;
    let restype = match &mut expr.details {
        ExprDetail::Var(x) => {
            let name = x.0;
            match ctx.table.search_variable(ctx.level, name.trie_index) {
                Some(x) => {
                    x.val_type
                },
                None => {
                    ctx.errors.push(AnalyzeError::VariableNotFound(name));
                    Type::Unit
                },
            }
        },
        ExprDetail::Block(b) => {
            let exprs = b.exprs.clone();
            ctx.with_new_level(|ctx| {
                let mut last_type = Type::Unit;
                for x in exprs {
                    last_type = analyze_expr(ctx, x);
                }
                last_type
            })
        },
        ExprDetail::Lit(x) => {
            x.get_type()
        },
        ExprDetail::FunctionCall(f) => {
            let expr_loc = expr.loc;
            let fun_name = f.name.0.trie_index;


            let argst = f.args.clone().iter()
                .copied()
                .map(|x| analyze_expr(ctx, x))
                .collect();
            let signature = FunctionSignature::of(fun_name, argst);
            match ctx.table.find_function(&signature) {
                Some(x) => {
                    let expr = &mut ctx.table.exprs[expr_id];
                    let f = match &mut expr.details {
                        ExprDetail::FunctionCall(x) => x,
                        _ => unreachable!(),
                    };
                    f.function_id = Some(x);
                    ctx.table.get_function(x).get_return_type()
                },
                None => {
                    let f = match &ctx.table.exprs[expr_id].details {
                        ExprDetail::FunctionCall(x) => x,
                        _ => unreachable!(),
                    };

                    let argst = f.args.clone().iter()
                        .copied()
                        .map(|x| analyze_expr(ctx, x))
                        .collect();

                    // TODO: print the real name, or at least pass down the name id and figure it out later
                    ctx.errors.push(AnalyzeError::FunctionNotFound(expr_loc, format!("{}", fun_name), argst));
                    Type::Unit
                },
            }
        },
        ExprDetail::If(x) => {
            fn analyze_if_block(ctx: &mut Ctx, x: &IfBlock) -> Type {
                let cond_type = analyze_expr(ctx, x.cond);
                if cond_type != Type::Bool {
                    let cond_loc = ctx.table.exprs[x.cond].loc;
                    ctx.errors.push(AnalyzeError::MismatchedType(cond_loc, Type::Bool, cond_type));
                }
                ctx.with_new_level(|ctx| analyze_expr(ctx, x.then))
            }
            let x = x.clone();

            let mut block_iter = x.blocks.iter();

            let first = block_iter.next().expect("If must have at least one condition");
            let first_type = analyze_if_block(ctx, first);

            for block in block_iter {
                let t =  analyze_if_block(ctx, block);
                if t != first_type {
                    let then = &ctx.table.exprs[block.then];
                    ctx.errors.push(AnalyzeError::MismatchedType(then.loc, first_type, t));
                }
            }

            if let Some(tail_id) = x.tail {
                let t = ctx.with_new_level(|ctx| analyze_expr(ctx, tail_id));
                if t != first_type {
                    let tail_loc = ctx.table.exprs[tail_id].loc;
                    ctx.errors.push(AnalyzeError::MismatchedType(tail_loc, first_type, t));
                }
            }
            first_type
        },
        ExprDetail::Assign(x) => {
            let expr = x.expr;
            let name = x.name.0;
            let rhs_type = analyze_expr(ctx, expr);
            match ctx.table.search_variable(ctx.level, name.trie_index) {
                None => {
                    ctx.errors.push(AnalyzeError::VariableNotFound(name));
                    Type::Unit
                },
                Some(x) => {
                    if x.val_type != rhs_type {
                        ctx.errors.push(AnalyzeError::MismatchedType(x.loc, x.val_type, rhs_type));
                    }
                    x.val_type
                },
            }
        },
        ExprDetail::Decl(x) => {
            let loc = expr.loc;
            let assign_expr = x.assign.expr;
            let decl_name = x.assign.name.0;
            let rhs_type = analyze_expr(ctx, assign_expr);
            ctx.table.register_variable(ctx.level, decl_name.trie_index, rhs_type, loc);
            rhs_type
        },
    };
    let expr = &mut ctx.table.exprs[expr_id];
    expr.res_type = Some(restype);
    restype
}

pub fn analyze_all(table: &mut AstData) -> Vec<AnalyzeError> {
    let mut errors = vec![];
    let mut ctx = Ctx {
        table,
        errors: &mut errors,
        level: 0,
    };
    let flen = ctx.table.functions.len();
    for funi in 0..flen {
        ctx.level = ctx.table.create_level(None);
        let x = match &mut ctx.table.functions[funi] {
            FunctionDefinition::Builtin(_) => continue,
            FunctionDefinition::Custom(f) => f,
        };
        let ret_type = x.ret_type.unwrap();
        let body = x.body;
        // Initial level
        for a in x.args.clone() {
            ctx.table.register_variable(ctx.level, (a.0).0.trie_index, a.1, (a.0).0.span);
        }
        let t = analyze_expr(&mut ctx, body);
        if t != ret_type {
            let body_loc = ctx.table.exprs[body].loc;
            ctx.errors.push(AnalyzeError::MismatchedType(body_loc, ret_type, t));
        }
    }
    errors
}