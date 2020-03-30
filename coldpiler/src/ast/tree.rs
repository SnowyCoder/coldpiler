
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Expr {
    Ident(Identifier),
    Block(Block),
    Lit(Value),
    Operation(Box<Expr>, Identifier, Box<Expr>),
    If(IfExpr),
    Print(Box<Expr>),
    Assign(Box<Assign>)
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Identifier(pub String);

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Block {
    pub exprs: Vec<BlockEntry>
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum BlockEntry {
    Expr(Expr),
    Decl(Declaration),
    Unit
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Value {
    Unit,
    I32(i32),
    Bool(bool)
}

impl Value {
    pub fn get_type(&self) -> Type {
        match self {
            Value::Unit => Type::Unit,
            Value::I32(_) => Type::I32,
            Value::Bool(_) => Type::Bool,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Value::I32(x) => Some(*x),
            _ => None
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(x) => Some(*x),
            _ => None
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Type {
    Unit, I32, Bool,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct IfExpr {
    pub blocks: Vec<IfBlock>,
    pub tail: Option<Block>
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct IfBlock {
    pub cond: Expr,
    pub then: Block,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Declaration {
    pub mutable: bool,
    pub assign: Assign,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Assign {
    pub name: Identifier,
    pub expr: Expr,
}

