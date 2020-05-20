use std::collections::HashMap;

use coldpiler_parser::scanner::TokenLoc;

use crate::ast::*;
use crate::context::Context;
use crate::symbol_table::{FunctionId, LevelId};

pub fn exec_function_builtin(fun: BuiltinFunction, args: Vec<Value>) -> Value {
    match fun {
        BuiltinFunction::I32Add => Value::I32(args[0].as_i32() + args[1].as_i32()),
        BuiltinFunction::I32Sub => Value::I32(args[0].as_i32() - args[1].as_i32()),
        BuiltinFunction::I32Mul => Value::I32(args[0].as_i32() * args[1].as_i32()),
        BuiltinFunction::I32Div => Value::I32(args[0].as_i32() / args[1].as_i32()),
        BuiltinFunction::I32Gt => Value::Bool(args[0].as_i32() > args[1].as_i32()),
        BuiltinFunction::I32Gte => Value::Bool(args[0].as_i32() >= args[1].as_i32()),
        BuiltinFunction::I32Lt => Value::Bool(args[0].as_i32() < args[1].as_i32()),
        BuiltinFunction::I32Lte => Value::Bool(args[0].as_i32() <= args[1].as_i32()),
        BuiltinFunction::I32Eq => Value::Bool(args[0].as_i32() == args[1].as_i32()),
        BuiltinFunction::I32Neq => Value::Bool(args[0].as_i32() != args[1].as_i32()),
        BuiltinFunction::BoolAnd => Value::Bool(args[0].as_bool() && args[1].as_bool()),
        BuiltinFunction::BoolOr => Value::Bool(args[0].as_bool() || args[1].as_bool()),
        BuiltinFunction::BoolEq => Value::Bool(args[0].as_bool() == args[1].as_bool()),
        BuiltinFunction::PrintlnI32 => {
            println!("{}", args[0].as_i32());
            Value::Unit
        } ,
        BuiltinFunction::PrintlnBool => {
            println!("{}", args[0].as_bool());
            Value::Unit
        },
    }
}

pub fn exec_function(table: &mut SymbolTable, fun: FunctionId, args: Vec<Value>)  -> Value {
    match &table.context.sym_table.functions[fun] {
        FunctionDefinition::Builtin(x) => exec_function_builtin(*x, args),
        FunctionDefinition::Custom(fun) => {
            let mut nt = SymbolTable::new(table.context);
            let level = table.context.sym_table.exprs[fun.body].level;
            for (atype, aval) in fun.args.iter().zip(args) {
                nt.assign(level, (atype.0).0, aval);
            }
            exec_expr(&mut nt, fun.body)
        },
    }
}

pub fn exec_if(table: &mut SymbolTable, expr: &IfExpr) -> Value {
    for x in expr.blocks.iter() {
        let cond = exec_expr(table, x.cond);
        let cond = cond.as_bool();

        if cond {
            return exec_expr(&mut table.clone(), x.then);
        }
    }
    expr.tail.as_ref().map_or(Value::Unit, |x| {
        exec_expr(&mut table.clone(), *x)
    })
}

pub fn exec_expr(mut table: &mut SymbolTable, expr_id: ExprId) -> Value {
    let expr = &table.context.sym_table.exprs[expr_id];
    match &expr.details {
        ExprDetail::Var(x) => table.query(expr.level, x.0).clone(),
        ExprDetail::Block(b) => {
            let mut last_val = Value::Unit;
            for entry in b.exprs.iter().copied() {
                last_val = exec_expr(&mut table, entry);
            }
            last_val
        },
        ExprDetail::Lit(l) => l.clone(),
        ExprDetail::If(x) => exec_if(table, x),
        ExprDetail::Assign(x) => {
            let val = exec_expr(table, x.expr);
            table.assign(expr.level, x.name.0, val)
        },
        ExprDetail::FunctionCall(f) => {
            let args: Vec<_> = f.args.iter().map(|x| exec_expr(table, *x)).collect();
            exec_function(table, f.function_id.unwrap(), args)
        }
        ExprDetail::Decl(d) => {
            let val = exec_expr(&mut table, d.assign.expr);
            let val = table.assign(expr.level, d.assign.name.0, val);
            val
        }
    }
}


#[derive(Clone)]
pub struct SymbolTable<'a> {
    context: &'a Context,
    // (level, name index) => Stored Value.
    table: HashMap<(LevelId, u32), Value>
}

impl<'a> SymbolTable<'a> {
    pub fn new(context: &'a Context) -> Self {
        SymbolTable {
            context,
            table: HashMap::new()
        }
    }

    fn assign(&mut self, level: LevelId, rhs: TokenLoc, val: Value) -> Value {
        let level = self.context.sym_table.search_variable(level, rhs.trie_index).unwrap().level;
        self.table.insert((level, rhs.trie_index), val.clone());
        val
    }

    fn query(&self, level: LevelId, name: TokenLoc) -> &Value {
        let level = self.context.sym_table.search_variable(level, name.trie_index).unwrap().level;
        self.table.get(&(level, name.trie_index)).unwrap()
    }
}