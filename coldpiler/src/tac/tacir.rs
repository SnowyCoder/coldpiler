use std::collections::HashMap;

use crate::ast::{BuiltinFunction, Declaration, ExprDetail, ExprId, FunctionDefinition, Type, Value};
use crate::ast::ast_data::{AstData, FunctionId, LevelId};
use core::fmt;
use std::cmp::{min, max};

// Three Address Code Intermediate Representation (TacIR)
// It's a lower level representation that resembles machine code
// this is used to fill the gap between the higher level AST and the lower level assembly.
// It still offers unlimited register usage and some abstract operations that the target machine
// might not support.

pub type VarId = u32;
type LabelId = u32;
pub type CodeLoc = u32;

#[derive(Clone, Debug)]
pub struct CFGNode {
    pub start: CodeLoc,
    pub end: CodeLoc,
    pub next: Vec<usize>,
    pub prev: Vec<usize>, // Back edge
    pub height: usize, // Min distance from the start of the function
    pub dominates: Vec<usize>,
    pub is_dominated: Vec<usize>,
    pub immediate_dominator: usize,

    // Stored as Map<res, Vec<(a, b)>> so that
    // res = PHI(a0, a1, a2...), and b0, b1, b2... is from which predecessor the parameter came.
    pub ssa_phi_origin: HashMap<u32, Vec<(u32, u32)>>
}

impl CFGNode {
    pub fn new(start: CodeLoc) -> Self {
        CFGNode {
            start,
            end: 0,
            next: vec![],
            prev: vec![],
            height: std::usize::MAX,
            dominates: vec![],
            is_dominated: vec![],
            immediate_dominator: std::usize::MAX,
            ssa_phi_origin: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TacFunction {
    pub param_count: u16,
    pub return_count: u16,
    pub is_leaf: bool,
    pub vars: Vec<TacVar>,
    pub instr: Vec<TacOp>,
    pub cfg: Vec<CFGNode>,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum Operation {
    Add,
    Sub,

    Or,
    And,
    Xor,

    Eq,
    Ne,
    Lt,
    Ltu,
    Ge,
    Geu
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum Operator {
    Var(VarId),
    Imm(i32),
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum EnvCall {
    PrintInt,
    PrintString,
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum TacOp {
    Label(LabelId),
    LoadImm(VarId, i32),
    PhiFun(VarId, Vec<VarId>),
    AssignOp(VarId, VarId, Operation, Operator),
    BranchIfZero(LabelId, VarId),
    Branch(LabelId),
    BindArg(u16, VarId),
    BindRet(u16, VarId),
    Call(FunctionId),
    EnvCall(EnvCall),
    Spill(VarId, u32),
    UnSpill(VarId, u32),
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum VarLen {
    Zero, Byte, Half, Word,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum VarType {
    Param(u16), Ret(u16), Custom,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct TacVar {
    pub definition: CodeLoc, // Only available in SSA form
    pub assignments: u32,
    pub usages: u32,
    pub vtype: VarType,
    pub length: VarLen,
    pub live_range: (CodeLoc, CodeLoc),
    pub ssa_original_id: u32, // If the var has been renamed because of SSA, this will hold it's original id
    pub is_zero: bool, // True if all of the assignments are always 0.
}

impl fmt::Display for TacFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "---TacFunction")?;
        writeln!(f, "| params: {}", self.param_count)?;
        writeln!(f, "| vars: {}", self.vars.len())?;
        for (i, var) in self.vars.iter().enumerate() {
            let t = match var.vtype {
                VarType::Param(p) => format!("P{}", p),
                VarType::Ret(r) => format!("R{}", r),
                VarType::Custom => "C".to_string(),
            };
            writeln!(f, "| {}- {} {:?} ({}, {}) def: {}, usages: {}, z: {}", i, t, var.length, var.live_range.0, var.live_range.1, var.definition, var.usages, var.is_zero)?;
        }
        writeln!(f, "| CFG:")?;
        for (cfgi, cfg) in self.cfg.iter().enumerate() {
            writeln!(f, "| {}- {}:{} {:?} DOMS: {:?}, IDOM: {}", cfgi, cfg.start, cfg.end, cfg.next, cfg.dominates, cfg.immediate_dominator)?;
        }
        writeln!(f, "| Code:")?;
        for (index, instr) in self.instr.iter().enumerate() {
            write!(f, "| {}> ", index)?;
            match instr {
                TacOp::Label(x) => writeln!(f, "L{}:", x)?,
                TacOp::LoadImm(to, imm) => writeln!(f, "x{} = {}", to, imm)?,
                TacOp::PhiFun(to, vars) => writeln!(f, "x{} = PHI({})", to, vars.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))?,
                TacOp::AssignOp(res, a, op, b) => {
                    let ops = match op {
                        Operation::Add => "+",
                        Operation::Sub => "-",
                        Operation::Or => "|",
                        Operation::And => "&",
                        Operation::Xor => "^",
                        Operation::Eq => "==",
                        Operation::Ne => "!=",
                        Operation::Lt => "<",
                        Operation::Ltu => "<l",
                        Operation::Ge => ">=",
                        Operation::Geu => ">=l",
                    };
                    let prefix = match b {
                        Operator::Var(_) => "x",
                        Operator::Imm(_) => "",
                    };
                    let imm = match b {
                        Operator::Var(x) => *x as i64,
                        Operator::Imm(x) => *x as i64,
                    };
                    writeln!(f, "x{} = x{} {} {}{}", res, a, ops, prefix, imm)?
                },
                TacOp::BranchIfZero(label, reg) => writeln!(f, "BIZ x{}, L{}", reg, label)?,
                TacOp::Branch(label) => writeln!(f, "B {}", label)?,
                TacOp::BindArg(arg, var) => writeln!(f, "Arg {} = x{}", arg, var)?,
                TacOp::BindRet(ret, var) => writeln!(f, "Ret {} = x{}", ret, var)?,
                TacOp::Call(x) => writeln!(f, "Call {}", x)?,
                TacOp::EnvCall(x) => writeln!(f, "EnvCall {:?}", x)?,
                TacOp::Spill(virt, loc) => writeln!(f, "Spill(x{} to {})", virt, loc)?,
                TacOp::UnSpill(virt, loc) => writeln!(f, "UnSpill(x{} from {}", virt, loc)?,
            }
        }

        Ok(())
    }
}

impl TacFunction {
    pub fn insert_instruction(&mut self, pos: CodeLoc, instr: TacOp) {
        use TacOp::*;
        match &instr {
            Label(_) | BranchIfZero(_, _) | Branch(_)  => panic!("Cannot insert cfg-changing instructions"),
            _ => {},
        }
        for cfgn in self.cfg.iter_mut() {
            if cfgn.start > pos {
                cfgn.start += 1;
            }
            if cfgn.end >= pos {
                cfgn.end += 1;
            }
        }
        for var in self.vars.iter_mut() {
            if var.live_range.0 >= pos {
                var.live_range.0 += 1;
            }
            if var.live_range.1 >= pos {
                var.live_range.1 += 1;
            }
            if var.definition >= pos {
                var.definition += 1;
            }
        }
        match &instr {
            LoadImm(v, val) => {
                let var = &mut self.vars[*v as usize];
                var.definition = min(var.definition, pos);
                var.assignments += 1;
                var.live_range.0 = min(var.live_range.0, pos);
                var.live_range.1 = max(var.live_range.1, pos);
                var.is_zero = *val == 0 && var.assignments == 1;
            },
            AssignOp(v, a, _, b) => {
                let var = &mut self.vars[*v as usize];
                var.definition = min(var.definition, pos);
                var.assignments += 1;
                var.live_range.0 = min(var.live_range.0, pos);
                var.live_range.1 = max(var.live_range.1, pos);
                var.is_zero = false;

                let var = &mut self.vars[*a as usize];
                var.usages += 1;
                var.live_range.0 = min(var.live_range.0, pos);
                var.live_range.1 = max(var.live_range.1, pos);
                if let Operator::Var(v) = b {
                    let var = &mut self.vars[*v as usize];
                    var.usages += 1;
                    var.live_range.0 = min(var.live_range.0, pos);
                    var.live_range.1 = max(var.live_range.1, pos);
                }
            },
            PhiFun(v, vars) => {
                let var = &mut self.vars[*v as usize];
                var.definition = min(var.definition, pos);
                var.assignments += 1;
                var.live_range.0 = min(var.live_range.0, pos);
                var.live_range.1 = max(var.live_range.1, pos);
                var.is_zero = false;
                for v in vars.iter().copied() {
                    let var = &mut self.vars[v as usize];
                    var.usages += 1;
                    var.live_range.0 = min(var.live_range.0, pos);
                    var.live_range.1 = max(var.live_range.1, pos);
                }
            },
            BindArg(_, v) | BindRet(_, v) | BranchIfZero(_, v) => {
                let var = &mut self.vars[*v as usize];
                var.usages += 1;
                var.live_range.0 = min(var.live_range.0, pos);
                var.live_range.1 = max(var.live_range.1, pos);
            },
            _ => {},
        }

        self.instr.insert(pos as usize, instr);
    }

    pub fn erase_instruction(&mut self, pos: CodeLoc) {
        use TacOp::*;
        match &self.instr[pos as usize] {
            Label(_) | BranchIfZero(_, _) | Branch(_)  => panic!("Cannot erase cfg-changing instructions"),
            _ => {},
        }
        self.instr.remove(pos as usize);
        for cfgn in self.cfg.iter_mut() {
            if cfgn.start > pos {
                cfgn.start -= 1;
            }
            if cfgn.end >= pos {
                cfgn.end -= 1;
            }
        }
        // TODO How should I update the live ranges?
    }

    pub fn next_variable_id(&self) -> u32 {
        self.vars.len() as u32
    }

    pub fn create_variable(&mut self, def_loc: u32, len: VarLen, ssa_orig_id: Option<u32>) -> u32 {
        let id = self.vars.len() as u32;
        self.vars.push(TacVar {
            definition: def_loc,
            assignments: 0,
            usages: 0,
            vtype: VarType::Custom,
            length: len,
            live_range: (0, 0),
            ssa_original_id: ssa_orig_id.unwrap_or(id),
            is_zero: false,
        });
        id
    }
}

fn convert_builtin(data: &mut ConversionData, table: &AstData, f: BuiltinFunction, args: &[ExprId]) -> Option<VarId> {
    match f {
        // Functions
        BuiltinFunction::PrintlnBool => {
            let arg0 = convert_expr(data, table, args[0]).unwrap();
            data.emit(TacOp::BindArg(0, arg0));
            data.emit(TacOp::EnvCall(EnvCall::PrintInt));
            return None;
        },
        BuiltinFunction::PrintlnI32 => {
            let arg0 = convert_expr(data, table, args[0]).unwrap();
            data.emit(TacOp::BindArg(0, arg0));
            data.emit(TacOp::EnvCall(EnvCall::PrintInt));
            return None;
        },
        // Short-circuit-able operations (bool and, bool or)
        BuiltinFunction::BoolAnd => {
            // TODO: optimize
            // out = arg0()
            // if (out) out = arg1();

            let arg0 = convert_expr(data, table, args[0]).unwrap();
            let labelend = data.allocate_future_label();
            let tmp_reg = data.allocate_tmp_variable(VarLen::Byte);
            data.emit(TacOp::AssignOp(tmp_reg, arg0, Operation::Eq, Operator::Imm(0)));
            data.emit(TacOp::BranchIfZero(labelend, tmp_reg));
            let arg1 = convert_expr(data, table, args[1]).unwrap();
            data.emit(TacOp::AssignOp(arg0, arg1, Operation::Add, Operator::Imm(0)));
            data.emit(TacOp::Label(labelend));
            return Some(arg0);
        },
        BuiltinFunction::BoolOr => {
            // out = arg0()
            // if (!out) out = arg1();
            let arg0 = convert_expr(data, table, args[0]).unwrap();
            let labelend = data.allocate_future_label();
            let tmp_reg = data.allocate_tmp_variable(VarLen::Byte);
            data.emit(TacOp::AssignOp(tmp_reg, arg0, Operation::Ne, Operator::Imm(0)));
            data.emit(TacOp::BranchIfZero(labelend, tmp_reg));
            let arg1 = convert_expr(data, table, args[1]).unwrap();
            data.emit(TacOp::AssignOp(arg0, arg1, Operation::Add, Operator::Imm(0)));
            data.emit(TacOp::Label(labelend));
            return Some(arg0);
        },
        _ => {}
    }

    // Everything else with two args and one return.

    let arg0 = convert_expr(data, table, args[0]).unwrap();
    let alen = data.vars[arg0 as usize].length;
    let arg1 = convert_expr(data, table, args[1]).unwrap();

    let out = match f {
        BuiltinFunction::I32Add => {
            let out = data.allocate_tmp_variable(alen);
            data.emit(TacOp::AssignOp(out, arg0, Operation::Add, Operator::Var(arg1)));
            out
        },
        BuiltinFunction::I32Sub => {
            let out = data.allocate_tmp_variable(alen);
            data.emit(TacOp::AssignOp(out, arg0, Operation::Sub, Operator::Var(arg1)));
            out
        },
        BuiltinFunction::I32Mul => {
            unimplemented!();// TODO
        },
        BuiltinFunction::I32Div => {
            unimplemented!();// TODO
        },
        BuiltinFunction::I32Gt => {
            let out = data.allocate_tmp_variable(alen);
            data.emit(TacOp::AssignOp(out, arg1, Operation::Lt, Operator::Var(arg0)));
            out
        },
        BuiltinFunction::I32Gte => {
            let out = data.allocate_tmp_variable(VarLen::Byte);
            data.emit(TacOp::AssignOp(out, arg0, Operation::Ge, Operator::Var(arg1)));
            out
        },
        BuiltinFunction::I32Lt => {
            let out = data.allocate_tmp_variable(VarLen::Byte);
            data.emit(TacOp::AssignOp(out, arg0, Operation::Lt, Operator::Var(arg1)));
            out
        },
        BuiltinFunction::I32Lte => {
            let out = data.allocate_tmp_variable(VarLen::Byte);
            data.emit(TacOp::AssignOp(out, arg1, Operation::Ge, Operator::Var(arg0)));
            out
        },
        BuiltinFunction::I32Eq => {
            let out = data.allocate_tmp_variable(VarLen::Byte);
            data.emit(TacOp::AssignOp(out, arg0, Operation::Eq, Operator::Var(arg1)));
            out
        },
        BuiltinFunction::I32Neq => {
            let out = data.allocate_tmp_variable(VarLen::Byte);
            data.emit(TacOp::AssignOp(out, arg0, Operation::Ne, Operator::Var(arg1)));
            out
        },
        BuiltinFunction::BoolEq => {
            let out = data.allocate_tmp_variable(VarLen::Byte);
            data.emit(TacOp::AssignOp(out, arg0, Operation::Eq, Operator::Var(arg1)));
            out
        },
        _ => panic!("Unhandled builtin function")
    };
    Some(out)
}

fn convert_expr(data: &mut ConversionData, table: &AstData, eid: ExprId) -> Option<VarId> {
    let expr = &table.exprs[eid];
    match &expr.details {
        ExprDetail::Var(x) => {
            let var = table.search_variable(expr.level, x.0.trie_index).unwrap();
            Some(data.get_variable_by_name(var.level, var.name))
        },
        ExprDetail::Block(x) => {
            let mut last_id = None;
            for ex in x.exprs.iter().copied() {
                last_id = convert_expr(data, &table, ex)
            }
            last_id
        },
        ExprDetail::Lit(x) => {
            let reg = data.allocate_tmp_variable(x.get_type().get_len());
            match x {
                Value::Unit => {},
                Value::I32(x) => {
                    data.emit(TacOp::LoadImm(reg, *x));
                },
                Value::Bool(b) => {
                    let v = if *b { 1 } else { 0 };
                    data.emit(TacOp::LoadImm(reg, v));
                },
            }
            Some(reg)
        },
        ExprDetail::FunctionCall(x) => {
            match &table.functions[x.function_id.unwrap()] {
                FunctionDefinition::Builtin(f) => convert_builtin(data, table, *f, &x.args),
                FunctionDefinition::Custom(f) => {
                    let args: Vec<_> = x.args.iter()
                        .map(|x| convert_expr(data, table, *x).unwrap())
                        .collect();


                    for (argi, argv) in args.iter().enumerate() {
                        data.emit(TacOp::BindArg(argi as u16, *argv))
                    }
                    data.emit(TacOp::Call(x.function_id.unwrap()));
                    if f.ret_type.unwrap() != Type::Unit {
                        let out_reg = data.allocate_tmp_variable(f.ret_type.unwrap().get_len());
                        data.emit(TacOp::BindRet(0, out_reg));
                        Some(out_reg)
                    } else {
                        None
                    }
                },
            }
        },
        ExprDetail::If(x) => {
            // if (cond_a) {
            //   code_a
            // } else if (cond_b) {
            //   code_b
            // ...
            // } else {
            //   code_else
            // }
            // can be linearized to:
            // if !cond_a() jmp to label_a
            // code_a
            // label_a
            // if !cond_b() jmp to label_b
            // code_b
            // label_b
            // ....
            // code_else
            let has_res = expr.res_type.unwrap() != Type::Unit;
            let out_reg = if has_res {
                Some(data.allocate_tmp_variable(expr.res_type.unwrap().get_len()))
            } else {
                None
            };
            for block in x.blocks.iter().copied() {
                let label = data.allocate_future_label();
                // is the unwrap safe? yes as the semantic analyzer would not have allowed an Unit
                // value as the if condition
                let cond_rel = convert_expr(data, table, block.cond).unwrap();
                let tmp_reg = data.allocate_tmp_variable(VarLen::Byte);
                data.emit(TacOp::AssignOp(tmp_reg, cond_rel, Operation::Eq, Operator::Imm(0)));
                data.emit(TacOp::BranchIfZero(label, tmp_reg));
                let reg_res = convert_expr(data, table, block.then);
                if let Some(out) = out_reg {
                    data.emit(TacOp::AssignOp(out, reg_res.unwrap(), Operation::Add, Operator::Imm(0)));
                }
                data.emit(TacOp::Label(label));
            }
            if let Some(tail) = x.tail {
                let reg_res = convert_expr(data, table, tail);
                if let Some(out) = out_reg {
                    data.emit(TacOp::AssignOp(out, reg_res.unwrap(), Operation::Add, Operator::Imm(0)));
                }
            }
            out_reg
        },
        ExprDetail::Assign(x) => {
            let rhs = convert_expr(data, table, x.expr).unwrap();


            let var = table.search_variable(expr.level, x.name.0.trie_index).unwrap();
            let reg = data.get_variable_by_name(var.level, var.name);
            data.emit(TacOp::AssignOp(reg, rhs, Operation::Add, Operator::Imm(0)));
            Some(reg)
        }
        ExprDetail::Decl(Declaration { assign: x, .. }) => {
            let t = table.exprs[x.expr].res_type.unwrap();
            let rhs = convert_expr(data, table, x.expr).unwrap();

            let reg = data.declare_variable(expr.level, x.name.0.trie_index, VarType::Custom, t.get_len());
            data.emit(TacOp::AssignOp(reg, rhs, Operation::Add, Operator::Imm(0)));
            Some(reg)
        }
    }
}

pub fn ast_function_to_tac_ir(table: &AstData, fun: FunctionId) -> TacFunction {
    let f = match &table.functions[fun] {
        FunctionDefinition::Builtin(_) => panic!("Cannot convert a builtin"),
        FunctionDefinition::Custom(x) => x,
    };

    let body = &table.exprs[f.body];

    let mut cdata = ConversionData::new();

    for (argi, arg) in f.args.iter().enumerate() {
        let vt = table.exprs[argi].res_type.unwrap().get_len();
        cdata.declare_variable(body.level, (arg.0).0.trie_index, VarType::Param(argi as u16), vt);
    }

    convert_expr(&mut cdata, &table, f.body);
    cdata.on_close();

    let has_return = body.res_type.unwrap() != Type::Unit;

    let mut fun = TacFunction {
        param_count: f.args.len() as u16,
        return_count: if has_return { 1 } else { 0 },
        is_leaf: false,
        vars: cdata.vars,
        instr: cdata.res,
        cfg: cdata.cfg,
    };
    fun.compute_domination();
    //fun.convert_to_ssa();
    fun
}

struct ConversionData {
    name_to_var: HashMap<(LevelId, u32), VarId>,
    vars: Vec<TacVar>,
    label_count: LabelId,
    label_to_cfg: HashMap<LabelId, usize>,
    next_instr_label: Option<LabelId>,
    res: Vec<TacOp>,
    cfg: Vec<CFGNode>,
    curr_cfg: usize,
}

// Problem: we cannot assign to the CFGs the real CFG indices, why is that?
// we need to be able to link CFGs that are not yet written, as example"
// if (...) { ... } else { ... }
// will be translated to
// BNE label_else
// but label_else hasn't been emitted yet, to solve this problem we initially just save
// label ids instead of CFG indices, then we replace them when the emit phase is complete.
impl ConversionData {
    fn new() -> Self {
        Self {
            name_to_var: HashMap::new(),
            vars: Vec::new(),
            label_count: 0,
            label_to_cfg: HashMap::new(),
            next_instr_label: None,
            res: Vec::new(),
            cfg: vec![CFGNode::new(0)],
            curr_cfg: 0
        }
    }

    fn declare_variable(&mut self, level: LevelId, name: u32, vtype: VarType, vlen: VarLen) -> VarId {
        let loc = self.res.len() as u32;
        let var_index = self.vars.len() as u32;
        self.vars.push(TacVar {
            definition: loc,
            assignments: 0,
            usages: 0,
            vtype,
            length: vlen,
            live_range: (loc, 0),
            ssa_original_id: var_index,
            is_zero: false,
        });
        self.name_to_var.insert((level, name), var_index);
        var_index
    }

    fn get_variable_by_name(&mut self, level: LevelId, name: u32) -> VarId {
        *self.name_to_var.get(&(level, name)).expect("Cannot find variable by name")
    }

    fn allocate_tmp_variable(&mut self, vlen: VarLen) -> VarId {
        let loc = self.res.len() as u32;
        let var_index = self.vars.len() as u32;
        self.vars.push(TacVar {
            definition: loc,
            assignments: 0,
            usages: 0,
            vtype: VarType::Custom,
            length: vlen,
            live_range: (loc, 0),
            ssa_original_id: var_index,
            is_zero: false,
        });
        var_index
    }

    fn allocate_future_label(&mut self) -> LabelId {
        let id = self.label_count;
        self.label_count += 1;
        id
    }

    fn get_current_label(&mut self) -> LabelId {
        if let Some(next_lab) = self.next_instr_label {
            return next_lab;
        }
        let label = self.allocate_future_label();
        self.emit(TacOp::Label(label));
        return label;
    }

    fn emit(&mut self, op: TacOp) {
        let curr_pc = self.res.len() as u32;
        self.res.push(op.clone());
        self.next_instr_label = None;
        match op {
            TacOp::LoadImm(v, val) => {
                let v = &mut self.vars[v as usize];
                v.assignments += 1;
                v.is_zero = val == 0 && v.assignments == 1;
            }
            TacOp::Label(labelid) => {
                let cfg_len = self.cfg.len();
                let curr = self.cfg.get_mut(self.curr_cfg).unwrap();
                if curr.start != curr_pc {
                    // Not empty
                    let nexti = cfg_len;
                    curr.end = curr_pc;
                    curr.next.push(labelid as usize);
                    self.cfg.push(CFGNode::new(curr_pc + 1));
                    self.curr_cfg = nexti;
                } else {
                    // Current CFG Empty
                    // ...
                    // label1:
                    // label2: <-- You are pushing this
                    curr.start += 1;
                }

                self.label_to_cfg.insert(labelid, self.curr_cfg);
                self.next_instr_label = Some(labelid);
            },
            TacOp::BranchIfZero(label, _) => {
                let after_label = self.allocate_future_label();
                let curr = self.cfg.get_mut(self.curr_cfg).unwrap();
                curr.end = curr_pc + 1;
                curr.next.push(label as usize);
                self.res.push(TacOp::Label(after_label));
                self.next_instr_label = Some(after_label);
                curr.next.push(after_label as usize);
                self.label_to_cfg.insert(after_label, self.cfg.len());
                self.cfg.push(CFGNode::new(curr_pc + 2));// curr_pc -> BZ, Label, _ <- next istr = pc +
                self.curr_cfg = self.cfg.len() - 1;
            },
            TacOp::Branch(label) => {
                let curr = self.cfg.get_mut(self.curr_cfg).unwrap();
                curr.end = curr_pc + 1;
                curr.next.push(label as usize);
                self.cfg.push(CFGNode::new(curr_pc + 1));
                self.curr_cfg = self.cfg.len() - 1;
            },
            _ => {}
        }
    }

    fn on_close(&mut self) {
        self.cfg[self.curr_cfg].end = self.res.len() as u32;
        for (cfg_ind, cfg) in self.cfg.iter_mut().enumerate() {
            // Replace label indices with CFG indices
            for x in cfg.next.iter_mut() {
                *x = *self.label_to_cfg.get(&(*x as u32)).unwrap();
            }
        }

        // Sometimes I hate the borrow checker...
        for cfg in 0..self.cfg.len() {
            for n in self.cfg[cfg].next.clone() {
                self.cfg.get_mut(n).unwrap().prev.push(cfg);
            }
        }
    }
}