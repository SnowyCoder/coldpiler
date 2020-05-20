use std::fmt;

use coldpiler_parser::loc::SpanLoc;
use coldpiler_parser::scanner::TokenLoc;
use coldpiler_util::radix_tree::RadixTree;

use crate::context::TokenId;
use crate::symbol_table::{FunctionId, LevelId, SymbolTable};

pub type ExprId = usize;

macro_rules! create_bank {
    ($name:ident, $($entry:ident=$entry_val:literal, )*) => {
        #[derive(Clone, Debug)]
        pub struct $name {
            $(pub $entry: TokenId,)*
        }

        impl $name {
            pub fn create(tree: &mut RadixTree<u8>) -> Self {
                $name {
                    $($entry: tree.insert($entry_val),)*
                }
            }
        }
    }
}

create_bank!(CommonIdentBank,
    add=b"+",
    sub=b"-",
    mul=b"*",
    div=b"/",
    gt =b">",
    gte=b">=",
    lt =b"<",
    lte=b"<=",
    eq =b"==",
    neq=b"!=",
    and=b"and",
    or =b"or",

    i32 =b"I32",
    bool=b"Bool",
    unit=b"Unit",

    main=b"main",
    println=b"println",
);


#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FunctionSignature {
    pub name: TokenId,
    pub args: Vec<Type>,
}

impl FunctionSignature {
    pub fn of(name: TokenId, args: Vec<Type>) -> Self {
        FunctionSignature {
            name, args
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum FunctionDefinition {
    Builtin(BuiltinFunction),
    Custom(FunctionDeclaration),
}

impl FunctionDefinition {
    pub fn create_signature(&self, bank: &CommonIdentBank) -> FunctionSignature {
        match self {
            FunctionDefinition::Builtin(x) => x.create_signature(bank),
            FunctionDefinition::Custom(x) => x.create_signature(),
        }
    }

    pub fn get_return_type(&self) -> Type {
        match self {
            FunctionDefinition::Builtin(x) => x.get_return_type(),
            FunctionDefinition::Custom(x) => x.ret_type.expect("Function w/no return type"),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum BuiltinFunction {
    I32Add,
    I32Sub,
    I32Mul,
    I32Div,
    I32Gt,
    I32Gte,
    I32Lt,
    I32Lte,
    I32Eq,
    I32Neq,
    BoolAnd,
    BoolOr,
    BoolEq,
    PrintlnI32,
    PrintlnBool,
}

impl BuiltinFunction {
    pub fn create_signature(&self, bank: &CommonIdentBank) -> FunctionSignature {
        use Type::*;
        use BuiltinFunction::*;
        match self {
            I32Add => FunctionSignature::of(bank.add, vec![I32, I32]),
            I32Sub => FunctionSignature::of(bank.sub, vec![I32, I32]),
            I32Mul => FunctionSignature::of(bank.mul, vec![I32, I32]),
            I32Div => FunctionSignature::of(bank.div, vec![I32, I32]),
            I32Gt  => FunctionSignature::of(bank.gt,  vec![I32, I32]),
            I32Gte => FunctionSignature::of(bank.gte, vec![I32, I32]),
            I32Lt  => FunctionSignature::of(bank.lt,  vec![I32, I32]),
            I32Lte => FunctionSignature::of(bank.lte, vec![I32, I32]),
            I32Eq  => FunctionSignature::of(bank.eq,  vec![I32, I32]),
            I32Neq => FunctionSignature::of(bank.neq, vec![I32, I32]),
            BoolAnd => FunctionSignature::of(bank.and, vec![Bool, Bool]),
            BoolOr => FunctionSignature::of(bank.or,  vec![Bool, Bool]),
            BoolEq => FunctionSignature::of(bank.eq,  vec![Bool, Bool]),
            PrintlnI32 => FunctionSignature::of(bank.println, vec![I32]),
            PrintlnBool => FunctionSignature::of(bank.println, vec![Bool]),
        }
    }

    pub fn get_return_type(&self) -> Type {
        use Type::*;
        use BuiltinFunction::*;
        match self {
            I32Add => I32,
            I32Sub => I32,
            I32Mul => I32,
            I32Div => I32,
            I32Gt  => Bool,
            I32Gte => Bool,
            I32Lt  => Bool,
            I32Lte => Bool,
            I32Eq  => Bool,
            I32Neq => Bool,
            BoolAnd => Bool,
            BoolOr => Bool,
            BoolEq => Bool,
            PrintlnI32 => Unit,
            PrintlnBool => Unit,
        }
    }

    pub fn register_all(bank: &CommonIdentBank, table: &mut SymbolTable) {
        use BuiltinFunction::*;
        for x in &[I32Add, I32Sub, I32Mul, I32Div, I32Gt, I32Gte, I32Lt, I32Lte, I32Eq, I32Neq, BoolAnd, BoolOr, BoolEq, PrintlnI32, PrintlnBool] {
            table.register_function(&bank, FunctionDefinition::Builtin(*x));
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FunctionDeclaration {
    pub name: Identifier,
    pub args: Vec<(Identifier, Type)>,
    pub body: ExprId,
    pub ret_type: Option<Type>,
}

impl FunctionDeclaration {
    pub fn create_signature(&self) -> FunctionSignature {
        FunctionSignature {
            name: self.name.0.trie_index,
            args: self.args.iter().map(|x| x.1).collect(),
        }
    }
}


#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Block {
    pub exprs: Vec<ExprId>
}

impl Block {
    pub fn get_result_type(&self, table: &SymbolTable) -> Option<Type> {
        return self.exprs.last()
            .map(|x| table.exprs[*x].res_type)
            .unwrap_or(Some(Type::Unit))
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Expr {
    pub loc: SpanLoc,
    pub level: LevelId,
    pub details: ExprDetail,
    pub res_type: Option<Type>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ExprDetail {
    Var(Identifier),
    Block(Block),
    Lit(Value),
    FunctionCall(FunctionCall),
    If(IfExpr),
    Assign(Assign),
    Decl(Declaration)
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Identifier(pub TokenLoc);


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

    pub fn as_i32(&self) -> i32 {
        match self {
            Value::I32(x) => *x,
            _ => panic!("Value is not i32")
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            Value::Bool(x) => *x,
            _ => panic!("Value is not bool")
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Unit => write!(f, "()"),
            Value::I32(x) => write!(f, "{}", x),
            Value::Bool(x) => write!(f, "{}", x),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Type {
    Unit,
    I32,
    Bool,
    Custom(TokenId),
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Type::Custom(x) => {
                return write!(f, "Custom({})", x);
            },
            Type::Unit => "Unit",
            Type::I32 => "I32",
            Type::Bool => "Bool",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct IfExpr {
    pub blocks: Vec<IfBlock>,
    pub tail: Option<ExprId>
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct IfBlock {
    pub cond: ExprId,
    pub then: ExprId,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Declaration {
    pub mutable: bool,
    pub assign: Assign,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Assign {
    pub name: Identifier,
    pub expr: ExprId,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FunctionCall {
    pub name: Identifier,
    pub args: Vec<ExprId>,
    pub function_id: Option<FunctionId>,
}

