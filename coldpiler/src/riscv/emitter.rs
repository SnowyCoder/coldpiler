use crate::tac::{TacProgram, TacFunction, TacOp, Operator, Operation, EnvCall, CodeLoc, VarId, VarLen};
use crate::riscv::instructions::*;
use std::collections::HashMap;
use crate::ast::ast_data::AstData;
use crate::ast::FunctionDefinition;
use crate::riscv::instructions::Instr::{JumpAndLink, ArithmeticImm};
use crate::register_allocator::{VariablesDisposition, VarAllocAction, RegisterConfig, SingleRegisterConfig, SavedBy, ConvRegType};
use std::env::var;

pub fn create_register_config() -> RegisterConfig {
    use SavedBy::*;
    use ConvRegType::*;
    RegisterConfig {
        regs: vec![
            SingleRegisterConfig::of( 5, Caller, None),
            SingleRegisterConfig::of( 6, Caller, None),
            SingleRegisterConfig::of( 7, Caller, None),
            SingleRegisterConfig::of( 8, Callee, None),
            SingleRegisterConfig::of( 9, Callee, None),
            SingleRegisterConfig::of(10, Caller, Argument),
            SingleRegisterConfig::of(11, Caller, Argument),
            SingleRegisterConfig::of(12, Caller, Argument),
            SingleRegisterConfig::of(13, Caller, Argument),
            SingleRegisterConfig::of(14, Caller, Argument),
            SingleRegisterConfig::of(15, Caller, Argument),
            SingleRegisterConfig::of(16, Caller, Argument),
            SingleRegisterConfig::of(17, Caller, Argument),
            SingleRegisterConfig::of(18, Callee, None),
            SingleRegisterConfig::of(19, Callee, None),
            SingleRegisterConfig::of(20, Callee, None),
            SingleRegisterConfig::of(21, Callee, None),
            SingleRegisterConfig::of(22, Callee, None),
            SingleRegisterConfig::of(23, Callee, None),
            SingleRegisterConfig::of(24, Callee, None),
            SingleRegisterConfig::of(25, Callee, None),
            SingleRegisterConfig::of(26, Callee, None),
            SingleRegisterConfig::of(27, Callee, None),
            SingleRegisterConfig::of(28, Caller, None),
            SingleRegisterConfig::of(29, Caller, None),
            SingleRegisterConfig::of(30, Caller, None),
            SingleRegisterConfig::of(31, Caller, None),
        ],
        static_zero: Some(0),
    }
}

fn emit_assign(data: &EmitData, r: VarId, a: VarId, op: Operation, b: Operator, res: &mut Vec<Instr>)  {
    // Load reg. in t1
    let rd = data.get_var(r);
    let p0 = data.get_var(a);

    if rd == 0 || (rd == p0 && b == Operator::Imm(0) && (op == Operation::Add || op == Operation::Or || op == Operation::Xor)) {
        return;
    }

    let p1 = match b {
        Operator::Var(x) => data.get_var(x),
        Operator::Imm(_) => 0u8,
    };

    match op {
        Operation::Add => {
            match b {
                Operator::Var(_) => { res.push(Instr::ArithmeticReg { rd, rs1: p0, rs2: p1, op: ArithOp::Add }) },
                Operator::Imm(x) => { res.push(Instr::ArithmeticImm { rd, rs1: p0, imm: x as i16, op: ArithImmOp::Add })},
            }
        },
        Operation::Sub => {
            match b {
                Operator::Var(_) => { res.push(Instr::ArithmeticReg { rd, rs1: p0, rs2: p1, op: ArithOp::Sub }) },
                Operator::Imm(x) => { res.push(Instr::ArithmeticImm { rd, rs1: p0, imm: -x as i16, op: ArithImmOp::Add })},
            }
        },
        Operation::Or => {
            match b {
                Operator::Var(_) => { res.push(Instr::ArithmeticReg { rd, rs1: p0, rs2: p1, op: ArithOp::Or }) },
                Operator::Imm(x) => { res.push(Instr::ArithmeticImm { rd, rs1: p0, imm: x as i16, op: ArithImmOp::Or })},
            }
        },
        Operation::And => {
            match b {
                Operator::Var(_) => { res.push(Instr::ArithmeticReg { rd, rs1: p0, rs2: p1, op: ArithOp::Add }) },
                Operator::Imm(x) => { res.push(Instr::ArithmeticImm { rd, rs1: p0, imm: x as i16, op: ArithImmOp::And})},
            }
        },
        Operation::Xor => {
            match b {
                Operator::Var(_) => { res.push(Instr::ArithmeticReg { rd, rs1: p0, rs2: p1, op: ArithOp::Xor }) },
                Operator::Imm(x) => { res.push(Instr::ArithmeticImm { rd, rs1: p0, imm: x as i16, op: ArithImmOp::Xor }) },
            }
        },
        Operation::Eq => {
            match b {
                Operator::Var(_) => { res.push(Instr::ArithmeticReg { rd: p0, rs1: p0, rs2: p1, op: ArithOp::Sub }) },
                Operator::Imm(x) => { res.push(Instr::ArithmeticImm { rd: p0, rs1: p0, imm: -x as i16, op: ArithImmOp::Add })},
            }
        },
        Operation::Ne => {
            match b {
                Operator::Var(_) => { res.push(Instr::ArithmeticReg { rd, rs1: p0, rs2: p1, op: ArithOp::Sub }) },
                Operator::Imm(x) => { res.push(Instr::ArithmeticImm { rd, rs1: p0, imm: -x as i16, op: ArithImmOp::Add })},
            }
            res.push(Instr::ArithmeticImm { rd: p0, rs1: p0, imm: 1, op: ArithImmOp::SetLessThanUnsigned, });
        },
        Operation::Lt => {
            match b {
                Operator::Var(_) => { res.push(Instr::ArithmeticReg { rd, rs1: p0, rs2: p1, op: ArithOp::SetLessThan }) },
                Operator::Imm(x) => { res.push(Instr::ArithmeticImm { rd, rs1: p0, imm: x as i16, op: ArithImmOp::SetLessThan })},
            }
        },
        Operation::Ltu => {
            match b {
                Operator::Var(_) => { res.push(Instr::ArithmeticReg { rd, rs1: p0, rs2: p1, op: ArithOp::SetLessThanUnsigned }) },
                Operator::Imm(x) => { res.push(Instr::ArithmeticImm { rd, rs1: p0, imm: x as i16, op: ArithImmOp::SetLessThanUnsigned })},
            }
        },
        Operation::Ge => {
            match b {
                Operator::Var(_) => { },
                Operator::Imm(x) => { res.push(Instr::ArithmeticImm { rd: p1, rs1: 0, imm: x as i16, op: ArithImmOp::Add })},
            }
            res.push(Instr::ArithmeticReg { rd, rs1: p0, rs2: p1, op: ArithOp::SetLessThan })
        },
        Operation::Geu => {
            match b {
                Operator::Var(_) => { },
                Operator::Imm(x) => { res.push(Instr::ArithmeticImm { rd: p1, rs1: 0, imm: x as i16, op: ArithImmOp::Add })},
            }
            res.push(Instr::ArithmeticReg { rd, rs1: p0, rs2: p1, op: ArithOp::SetLessThanUnsigned })
        },
    }
}

fn emit_fun(mut data: EmitData) -> Vec<Instr> {
    let mut label_map = HashMap::<u32, usize>::new();
    let mut jumps_to_fix = Vec::new();

    // Write the instructions using placeholders instead of labels
    // (we don't know how many instructions we could generate so we write the label index instead
    // of the jump difference, meanwhile we fill the label_map that will help us to reconstruct
    // the actual branches)
    let mut res = Vec::<Instr>::new();

    // Todo: handle >25565 arguments
    res.push(Instr::ArithmeticImm { rd: register_stack_pointer(), rs1: register_stack_pointer(), imm: -(data.tac_fun.vars.len() as i16) * 4 - 4, op: ArithImmOp::Add });
    res.push(Instr::Store { from: register_return_address(), to_reg: register_stack_pointer(), to_imm: -(data.tac_fun.vars.len() as i16) * 4, dtype: DataType::Word });


    data.prepare_next_instr(&mut res);
    for x in data.tac_fun.instr.iter().cloned() {
        match x {
            TacOp::Label(x) => {
                label_map.insert(x, res.len());
            },
            TacOp::LoadImm(var, imm) => {
                let rd = data.get_var(var);
                if rd != 0 {
                    res.push(Instr::ArithmeticImm {
                        rd,
                        rs1: 0,
                        imm: imm as i16,
                        op: ArithImmOp::Add
                    });
                }
            },
            TacOp::AssignOp(res_loc, a, op, b) => {
                emit_assign(&data, res_loc, a, op, b, &mut res);
            },
            TacOp::BranchIfZero(label, ir) => {
                // TODO: optimize
                let reg = data.get_var(ir);
                res.push(Instr::CondBranch { diff: label as i16, rs1: reg, rs2: 0, op: CondOp::Eq });
            },
            TacOp::Branch(l) => {
                // Exception: unconditioned branch, this should be done with JALs
                // Problem is we should use the palceholder method on this label too
                jumps_to_fix.push(res.len());
                res.push(JumpAndLink { link_reg: 0, label: l });
            },
            TacOp::BindArg(param_num, reg) => {
                // res.push(Instr::Load { to: register_argument(param_num as u8).expect("Invalid param num"), from_reg: register_stack_pointer(), from_imm: (reg as i16 - fun.var_count as i16) * 4, dtype: DataType::Word });
                let r = data.get_var(reg);
                res.push(Instr::ArithmeticImm { rd: register_argument(param_num as u8).expect("Invalid param num"), rs1: r, imm: 0, op: ArithImmOp::Add });
            },
            TacOp::BindRet(ret_num, reg) => {
                //res.push(Instr::Load { to: register_argument(ret_num as u8).expect("Invalid ret num"), from_reg: register_stack_pointer(), from_imm: (reg as i16 - fun.var_count as i16) * 4, dtype: DataType::Word});
                let r = data.get_var(reg);
                res.push(Instr::ArithmeticImm { rd: register_argument(ret_num as u8).expect("Invalid ret num"), rs1: r, imm: 0, op: ArithImmOp::Add });
            },
            TacOp::Call(x) => {
                let label = match &data.ast.functions[x] {
                    FunctionDefinition::Builtin(_) => panic!("Trying to jump to builtin function (?)"),
                    FunctionDefinition::Custom(x) => {
                        x.tac_id as u32
                    },
                };
                res.push(Instr::JumpAndLink { link_reg: register_return_address(), label });
            },
            TacOp::EnvCall(x) => {
                let par = register_argument(7).unwrap();
                let eindex = match x {
                    EnvCall::PrintInt => 1,
                    EnvCall::PrintString => 4,
                };
                res.push(Instr::ArithmeticImm {
                    rd: par, rs1: 0, imm: eindex, op: ArithImmOp::Add
                });
                res.push(Instr::EnvCall);
            },
            TacOp::PhiFun(_, _) => {
                unimplemented!();
            }
            TacOp::Spill(_, _) | TacOp::UnSpill(_, _) => panic!("Cannot emit spill/unspill"), // Should've been replaced in the register allocator phase
        }
        data.prepare_next_instr(&mut res);
    }

    res.push(Instr::Load { to: register_return_address(), from_reg: register_stack_pointer(), from_imm: -(data.tac_fun.vars.len() as i16) * 4, dtype: DataType::Word });
    res.push(Instr::ArithmeticImm { rd: register_stack_pointer(), rs1: register_stack_pointer(), imm: data.tac_fun.vars.len() as i16 * 4 + 4, op: ArithImmOp::Add });
    res.push(Instr::JumpAndLinkRegister { link_reg: 0, jump_reg: register_return_address(), label: 0 });

    // Adjust labels
    for (instr_index, instr) in res.iter_mut().enumerate() {
        match instr {
            Instr::CondBranch { diff, rs1, rs2, op } => {
                let dest = *label_map.get(&(*diff as u32)).expect("Invalid label") as u16;
                *diff = dest as i16 - instr_index as i16;
            },
            _ => {}
        }
    }

    for jump in jumps_to_fix.drain(..) {
        match res.get_mut(jump).unwrap() {
            JumpAndLink { link_reg, label } => {
                let dest = *label_map.get(&(*label as u32)).expect("Invalid label") as u32;
                *label = dest - jump as u32;
            }
            _ => unreachable!(),// We only push JALs to the 'jumps_to_fix' list
        }
    }

    res
}

struct EmitData<'a> {
    tac: &'a TacProgram,
    tac_fun: &'a TacFunction,
    ast: &'a AstData,
    current_tac_pc: CodeLoc,
    disposition: VariablesDisposition,
    var_alloc: HashMap<VarId, u8>,
}

impl EmitData<'_> {

    pub fn prepare_next_instr(&mut self, res: &mut Vec<Instr>) {
        //eprintln!("--Advance tac!");
        for x in self.disposition.advance_code() {
            match x {
                VarAllocAction::Alloc(var, reg, spill) => {
                    //eprintln!("|Alloc {} {} {:?}", var, reg, spill);
                    if let Some(x) = spill {
                        let len = match self.tac_fun.vars[var as usize].length {
                            VarLen::Zero => panic!("Invalid var length"),
                            VarLen::Byte => DataType::Byte,
                            VarLen::Half => DataType::Half,
                            VarLen::Word => DataType::Word,
                        };
                        // TODO: handle unsigned
                        res.push(Instr::Load {
                            to: reg as u8,
                            from_reg: register_stack_pointer(),
                            from_imm: (var as i16 - self.tac_fun.vars.len() as i16) * 4,
                            dtype: len,
                        });
                    }
                    self.var_alloc.insert(var, reg as u8);
                },
                VarAllocAction::DeAlloc(var, reg, spill) => {
                    //eprintln!("|DeAlloc {} {} {:?}", var, reg, spill);
                    if let Some(x) = spill {
                        let len = match self.tac_fun.vars[var as usize].length {
                            VarLen::Zero => panic!("Invalid var length"),
                            VarLen::Byte => DataType::Byte,
                            VarLen::Half => DataType::Half,
                            VarLen::Word => DataType::Word,
                        };
                        res.push(Instr::Store {
                            from: reg as u8,
                            to_reg: register_stack_pointer(),
                            to_imm: (var as i16 - self.tac_fun.vars.len() as i16) * 4,
                            dtype: len,
                        });
                    }
                    self.var_alloc.insert(var, reg as u8);
                },
            }
        }
    }

    pub fn get_var(&self, var_id: VarId) -> u8 {
        *self.var_alloc.get(&var_id).expect("")
    }
}


pub fn emit_program(tac: &TacProgram, ast: &AstData) -> Vec<Vec<Instr>> {
    let mut res = Vec::new();
    let config = create_register_config();
    for fun in tac.funs.iter() {
        let disposition = config.allocate(fun);
        let data = EmitData {
            tac,
            tac_fun: fun,
            ast,
            current_tac_pc: 0,
            disposition,
            var_alloc: HashMap::new(),
        };
        res.push(emit_fun(data));
    }
    res
}