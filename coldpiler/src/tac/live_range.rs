use crate::tac::{TacFunction, TacOp, TacVar, CodeLoc, Operator};
use std::cmp::{min, max};

impl TacFunction {
    pub fn recompute_live_ranges(&mut self) {
        let len = self.instr.len() as u32;
        for x in self.vars.iter_mut() {
            x.live_range = (len, 0);
        }

        let update_range = |pos: CodeLoc, v: u32, vars: &mut Vec<TacVar>| {
            let var = &mut vars[v as usize];
            var.live_range.0 = min(var.live_range.0, pos);
            var.live_range.1 = max(var.live_range.1, pos);
        };

        for (pos, i) in self.instr.iter().enumerate() {
            let pos = pos as CodeLoc;
            match i {
                TacOp::Label(_) => {},
                TacOp::LoadImm(v, _) | TacOp::BranchIfZero(_, v) | TacOp::BindArg(_, v) |
                TacOp::BindRet(_, v) | TacOp::Spill(v, _) | TacOp::UnSpill(v, _) => {
                    update_range(pos, *v, &mut self.vars);
                },
                TacOp::PhiFun(r, vars) => {
                    update_range(pos, *r, &mut self.vars);
                    for v in vars {
                        update_range(pos, *v, &mut self.vars);
                    }
                },
                TacOp::AssignOp(r, a, _, b) => {
                    update_range(pos, *r, &mut self.vars);
                    update_range(pos, *a, &mut self.vars);
                    if let Operator::Var(x) = b {
                        update_range(pos, *x, &mut self.vars);
                    }
                },
                TacOp::Branch(_) => {},
                TacOp::Call(_) => {},
                TacOp::EnvCall(_) => {},
            }
        }

    }
}
