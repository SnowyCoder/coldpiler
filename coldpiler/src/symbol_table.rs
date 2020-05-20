use std::collections::HashMap;
use std::slice::IterMut;

use coldpiler_parser::loc::SpanLoc;

use crate::ast::{Expr, ExprId, FunctionDefinition, FunctionSignature, Type};
use crate::ast::CommonIdentBank;
use crate::context::TokenId;

pub type FunctionId = usize;
pub type LevelId = usize;

#[derive(Clone)]
pub struct SymbolTable {
    pub functions: Vec<FunctionDefinition>,
    functions_by_signature: HashMap<FunctionSignature, FunctionId>,

    pub exprs: Vec<Expr>,
    variables: HashMap<(LevelId, TokenId), VariableDeclaration>,

    level_parent: Vec<LevelId>,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            functions: Vec::new(),
            functions_by_signature: HashMap::new(),
            exprs: Vec::new(),
            variables: HashMap::new(),
            level_parent: Vec::new(),
        }
    }

    pub fn register_function(&mut self, bank: &CommonIdentBank, function: FunctionDefinition) {
        let signature = function.create_signature(bank);
        self.functions.push(function);
        self.functions_by_signature.insert(signature, self.functions.len() - 1);
    }

    pub fn find_function(&self, signature: &FunctionSignature) -> Option<FunctionId> {
        self.functions_by_signature.get(signature).copied()
    }

    pub fn get_function(&self, fun_id: FunctionId) -> &FunctionDefinition {
        &self.functions[fun_id]
    }

    pub fn function_iter_mut(&mut self) -> IterMut<'_, FunctionDefinition> {
        self.functions.iter_mut()
    }

    pub fn create_level(&mut self, parent: Option<LevelId>) -> LevelId {
        let level = self.level_parent.len() as LevelId;
        self.level_parent.push(parent.unwrap_or(level));
        //eprintln!("Create Level: {} par: {:?}", level, parent);
        level
    }

    pub fn get_level_parent(&self, level: LevelId) -> Option<LevelId> {
        let par = self.level_parent[level];
        if par != level {
            Some(par)
        } else {
            None
        }
    }

    pub fn register_variable(&mut self, level: LevelId, name: TokenId, vtype: Type, loc: SpanLoc) {
        //eprintln!("Register var: {} at {:?} level {}", name, loc, level);
        let declaration = VariableDeclaration {
            level, name, loc,
            val_type: vtype
        };
        self.variables.insert((level, name), declaration);
    }

    pub fn search_variable(&self, mut level: LevelId, name: TokenId) -> Option<&VariableDeclaration> {
        //eprintln!("Find var: {} level {}", name, level);
        loop {
            if let Some(var) = self.variables.get(&(level, name)) {
                return Some(var);
            }
            if let Some(par) = self.get_level_parent(level) {
                level = par;
            } else {
                return None;
            }
        }
    }

    pub fn find_variable_exact(&self, level: LevelId, name: TokenId) -> Option<&VariableDeclaration> {
        self.variables.get(&(level, name))
    }

    pub fn register_expr(&mut self, expr: Expr) -> ExprId {
        self.exprs.push(expr);
        self.exprs.len() - 1
    }
}

#[derive(Clone)]
pub struct VariableDeclaration {
    pub level: LevelId,
    pub name: TokenId,
    pub loc: SpanLoc,
    pub val_type: Type,
}
