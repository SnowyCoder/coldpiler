use crate::ast::*;
use std::collections::HashMap;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InterpretError {
    AssignmentVarNotDeclared,
    AssignmentTypeError(Type, Type),
    VariableMotFound,
    OperationNotFound(Type, String, Type),
    IfConditionNotBool,
}

pub fn exec_operation(table: &mut SymbolTable, lhs: &Expr, op: &Identifier, rhs: &Expr) -> Result<Value, InterpretError> {
    let lhs = exec_expr(table, lhs)?;
    let rhs = exec_expr(table, rhs)?;
    // There should be a Map<(Type, Id, Type), Func> with all the possible functions,
    // for now it's hard-coded
    let res = match (lhs.clone(), op.0.as_str(), rhs.clone()) {
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
        None => Err(InterpretError::OperationNotFound(lhs.get_type(), op.0.clone(), rhs.get_type())),
    }
}

pub fn exec_if(table: &mut SymbolTable, expr: &IfExpr) -> Result<Value, InterpretError> {
    for x in expr.blocks.iter() {
        let cond = exec_expr(table, &x.cond)?;
        let cond = cond.as_bool().ok_or(InterpretError::IfConditionNotBool)?;

        if cond {
            return exec_block(table.clone(), &x.then);
        }
    }
    return expr.tail.as_ref().map_or(Ok(Value::Unit), |x| {
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
    match expr {
        Expr::Ident(x) => table.query(&x.0).map(|x| x.clone()),
        Expr::Block(b) => exec_block(table.clone(), b),
        Expr::Lit(l) => Ok(l.clone()),
        Expr::Operation(lhs, op, rhs) => exec_operation(table, lhs, op, rhs),
        Expr::If(x) => exec_if(table, x),
        Expr::Print(x) => exec_print(table, x),
        Expr::Assign(x) => {
            let val = exec_expr(table, &x.expr)?;
            table.assign(x.name.0.clone(), val)
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
                table.declare_assign(d.assign.name.0.clone(), val)
            },
            BlockEntry::Unit => Value::Unit,
        }
    }
    Ok(last_expr)
}


#[derive(Debug, Default, Clone)]
pub struct SymbolTable {
    table: HashMap<String, Value>
}

impl SymbolTable {
    fn declare_assign(&mut self, rhs: String, val: Value) -> Value{
        self.table.insert(rhs, val.clone());
        val
    }

    fn assign(&mut self, rhs: String, val: Value) -> Result<Value, InterpretError> {
        let old_val = match self.table.get(&rhs) {
            None => return Err(InterpretError::AssignmentVarNotDeclared),
            Some(x) => x,
        };
        if old_val.get_type() != val.get_type() {
            return Err(InterpretError::AssignmentTypeError(old_val.get_type(), val.get_type()))
        }
        self.table.insert(rhs, val.clone());
        Ok(val)
    }

    fn query(&self, name: &str) -> Result<&Value, InterpretError> {
        match self.table.get(name) {
            Some(x) => Ok(x),
            None => Err(InterpretError::VariableMotFound),
        }
    }
}