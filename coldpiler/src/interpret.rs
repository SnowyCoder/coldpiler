use crate::ast::*;
use std::collections::HashMap;
use crate::context::Context;
use crate::error::{CompilationError, ErrorLoc};
use coldpiler_parser::loc::SpanLoc;
use coldpiler_parser::scanner::TokenLoc;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InterpretError {
    AssignmentVarNotDeclared(SpanLoc),
    AssignmentTypeError(SpanLoc, Type, Type),
    VariableNotFound(SpanLoc),
    OperationNotFound(SpanLoc, Type, String, Type),
    IfConditionNotBool(SpanLoc),
}


impl CompilationError for InterpretError {
    fn error_type(&self) -> String {
        "Interpreter error".to_string()
    }

    fn loc(&self) -> ErrorLoc {
        ErrorLoc::SingleLocation(*match self {
            InterpretError::AssignmentVarNotDeclared(x) => x,
            InterpretError::AssignmentTypeError(x, _, _) => x,
            InterpretError::VariableNotFound(x) => x,
            InterpretError::OperationNotFound(x, _, _, _) => x,
            InterpretError::IfConditionNotBool(x) => x,
        })
    }

    fn summarize(&self) -> String {
        match self {
            InterpretError::AssignmentVarNotDeclared(_) => "Assigned variable not declared".to_string(),
            InterpretError::AssignmentTypeError(_, prev, new) => format!("Cannot assign value {} to variable of type {}", new, prev),
            InterpretError::VariableNotFound(_) => "Cannot find variable".to_string(),
            InterpretError::OperationNotFound(_, a, op, b) => format!("Operation {} not found for types {}, {}", op, a, b),
            InterpretError::IfConditionNotBool(_) => "If condition is not bool".to_string(),
        }
    }

    fn description(&self) -> String {
        match self {
            InterpretError::AssignmentVarNotDeclared(_) => "Create the variable before assigning it".to_string(),
            InterpretError::AssignmentTypeError(_, prev, new) => format!("Cannot assign value {} to variable of type {}, Create a new variable or change the types", new, prev),
            InterpretError::VariableNotFound(_) => "Cannot find variable".to_string(),
            InterpretError::OperationNotFound(_, a, op, b) => format!("Operation {} not found for types {}, {}. Did you forget to define one?", op, a, b),
            InterpretError::IfConditionNotBool(_) => "If condition is not bool".to_string(),
        }
    }
}
// TODO: implement CompilerError

pub fn exec_operation(table: &mut SymbolTable, lhs: &Expr, op: &Identifier, rhs: &Expr) -> Result<Value, InterpretError> {
    let lhs = exec_expr(table, lhs)?;
    let rhs = exec_expr(table, rhs)?;
    // There should be a Map<(Type, Id, Type), Func> with all the possible functions,
    // for now it's hard-coded
    let res = match (lhs.clone(), table.context.get_text(op.0.trie_index).as_str(), rhs.clone()) {
        (Value::I32(x), "+", Value::I32(y)) => {
            Some(Value::I32(x + y))
        },
        (Value::I32(x), "-", Value::I32(y)) => {
            Some(Value::I32(x - y))
        },
        (Value::I32(x), "*", Value::I32(y)) => {
            Some(Value::I32(x * y))
        },
        (Value::I32(x), "/", Value::I32(y)) => {
            Some(Value::I32(x / y))
        },
        (Value::I32(x), ">", Value::I32(y)) => {
            Some(Value::Bool(x > y))
        },
        (Value::I32(x), ">=", Value::I32(y)) => {
            Some(Value::Bool(x >= y))
        },
        (Value::I32(x), "<", Value::I32(y)) => {
            Some(Value::Bool(x < y))
        },
        (Value::I32(x), "<=", Value::I32(y)) => {
            Some(Value::Bool(x <= y))
        },
        (Value::I32(x), "==", Value::I32(y)) => {
            Some(Value::Bool(x == y))
        },
        (Value::I32(x), "!=", Value::I32(y)) => {
            Some(Value::Bool(x != y))
        },
        (Value::Bool(x), "and", Value::Bool(y)) => {
            Some(Value::Bool(x && y))
        },
        (Value::Bool(x), "or", Value::Bool(y)) => {
            Some(Value::Bool(x || y))
        },
        (Value::Bool(x), "==", Value::Bool(y)) => {
            Some(Value::Bool(x == y))
        },
        _ => None
    };
    match res {
        Some(x) => Ok(x),
        None => Err(InterpretError::OperationNotFound(op.0.span, lhs.get_type(), table.context.get_text(op.0.trie_index), rhs.get_type())),
    }
}

pub fn exec_if(table: &mut SymbolTable, expr: &IfExpr) -> Result<Value, InterpretError> {
    for x in expr.blocks.iter() {
        let cond = exec_expr(table, &x.cond)?;
        let cond = cond.as_bool().ok_or(InterpretError::IfConditionNotBool(x.cond.0))?;

        if cond {
            return exec_block(table.clone(), &x.then);
        }
    }
    expr.tail.as_ref().map_or(Ok(Value::Unit), |x| {
        exec_block(table.clone(), &x)
    })
}

pub fn exec_print(table: &mut SymbolTable, prt: &Expr) -> Result<Value, InterpretError> {
    let res = exec_expr(table, prt)?;
    match res {
        Value::Unit => println!("Unit"),
        Value::I32(x) => println!("{}", x),
        Value::Bool(x) => println!("{}", x),
    }
    Ok(Value::Unit)
}

pub fn exec_expr(table: &mut SymbolTable, expr: &Expr) -> Result<Value, InterpretError> {
    match &expr.1 {
        ExprDetail::Ident(x) => table.query(x.0).map(|x| x.clone()),
        ExprDetail::Block(b) => exec_block(table.clone(), b),
        ExprDetail::Lit(l) => Ok(l.clone()),
        ExprDetail::Operation(lhs, op, rhs) => exec_operation(table, lhs, op, rhs),
        ExprDetail::If(x) => exec_if(table, x),
        ExprDetail::Print(x) => exec_print(table, x),
        ExprDetail::Assign(x) => {
            let val = exec_expr(table, &x.expr)?;
            table.assign(x.name.0, val)
        },
    }
}

pub fn exec_block(mut table: SymbolTable, block: &Block) -> Result<Value, InterpretError> {
    let mut last_expr = Value::Unit;
    for entry in block.exprs.iter() {
        last_expr = match entry {
            BlockEntry::Expr(x) => exec_expr(&mut table, x)?,
            BlockEntry::Decl(d) => {
                let val =  exec_expr(&mut table, &d.assign.expr)?;
                table.declare_assign(d.assign.name.0, val)
            },
            BlockEntry::Unit => Value::Unit,
        }
    }
    Ok(last_expr)
}

pub fn exec(table: SymbolTable, block: &Block) -> Result<Value, ()> {
    let ctx = table.context;
    match exec_block(table, block) {
        Ok(x) => Ok(x),
        Err(e) => {
            ctx.print_error(&e);
            Err(())
        },
    }
}


#[derive(Clone)]
pub struct SymbolTable<'a> {
    context: &'a Context,
    // not a direct name to value hashmap, but a trie_index to value (should be faster?)
    table: HashMap<u32, Value>
}

impl<'a> SymbolTable<'a> {

    pub fn new(context: &'a Context) -> Self {
        SymbolTable {
            context,
            table: HashMap::new(),
        }
    }

    fn declare_assign(&mut self, rhs: TokenLoc, val: Value) -> Value{
        self.table.insert(rhs.trie_index, val.clone());
        val
    }

    fn assign(&mut self, rhs: TokenLoc, val: Value) -> Result<Value, InterpretError> {
        let old_val = match self.table.get(&rhs.trie_index) {
            None => return Err(InterpretError::AssignmentVarNotDeclared(rhs.span)),
            Some(x) => x,
        };
        if old_val.get_type() != val.get_type() {
            return Err(InterpretError::AssignmentTypeError(rhs.span, old_val.get_type(), val.get_type()))
        }
        self.table.insert(rhs.trie_index, val.clone());
        Ok(val)
    }

    fn query(&self, name: TokenLoc) -> Result<&Value, InterpretError> {
        match self.table.get(&name.trie_index) {
            Some(x) => Ok(x),
            None => Err(InterpretError::VariableNotFound(name.span)),
        }
    }
}