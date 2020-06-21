
// Static single assignment

use crate::tac::{TacFunction, TacOp, Operator, Operation};
use std::collections::{HashSet, HashMap};
use core::mem;

#[derive(Clone)]
struct BlockData {
    input_vars: HashMap<u32, u32>,
    output_vars: HashMap<u32, u32>,
    conflicts: HashMap<u32, HashSet<u32>>,
}

impl BlockData {
    fn new() -> BlockData {
        BlockData {
            input_vars: Default::default(),
            output_vars: Default::default(),
            conflicts: Default::default(),
        }
    }
}

impl TacFunction {

    pub fn convert_to_ssa(&mut self) {
        /*let mut dominance_frontier = Vec::new();
        dominance_frontier.resize_with(self.cfg.len(), || HashSet::new());

        for (cfg_node_i, cfg_node) in self.cfg.iter().enumerate() {
            if cfg_node.prev.len() < 2 {
                continue
            }
            for p in cfg_node.prev {
                let mut runner = p;

                while runner != cfg_node.immediate_dominator {
                    dominance_frontier[runner].insert(cfg_node_i);
                    runner = self.cfg[runner].immediate_dominator;
                }
            }
        }*/
        // TODO: this is NOT minimal!
        // We should use the dominance frontier, (algorithm commented above)
        // I'm too stupid/sleepy to apply it now, go on future me! I'm counting on you.


        pub fn apply_rename_map(var_map: &HashMap<u32, u32>, usage: &mut u32) {
            if let Some(new_name) = var_map.get(usage) {
                *usage = *new_name;
            }
        }

        let mut cfg_vars = vec![BlockData::new(); self.cfg.len()];

        // First, put in cfg_vars the declarations in each block (and disambiguate them with renames)
        // We also rename local variable usage, so that each block uses the original variables only
        // when their definition is inherited.
        // Ex:
        // A = A + 1
        // A = A + B
        // Becomes:
        // A1 = A + 1
        // A2 = A1 + B
        // output_vars: {A: A2}


        for cfgi in 0..self.cfg.len() {
            let start = self.cfg[cfgi].start;
            let end = self.cfg[cfgi].end;

            let mut var_map = &mut cfg_vars[cfgi];
            let mut current_rename = HashMap::new();

            for i in start..end {
                let op = &mut self.instr[i as usize];
                // First: rename operands
                match op {
                    TacOp::AssignOp(res, a, _, b) => {
                        apply_rename_map(&current_rename, a);

                        match b {
                            Operator::Var(mut b) => {
                                apply_rename_map(&current_rename, &mut b);
                            },
                            Operator::Imm(_) => {},
                        }
                    },
                    TacOp::BranchIfZero(_, v) | TacOp::BindArg(_, v) | TacOp::BindRet(_, v)  => {
                        apply_rename_map(&current_rename, v);
                    },
                    _ => {},
                }
                // Then: rename result
                match op {
                    TacOp::LoadImm(v, _) | TacOp::AssignOp(v, _, _, _) => {
                        let old_def = self.vars[*v as usize].definition;
                        if old_def != i {
                            // Variable already used
                            let v_val = *v;
                            let new_var = self.create_variable(i, self.vars[v_val as usize].length, Some(v_val));
                            var_map.output_vars.insert(v_val, new_var);
                            current_rename.insert(v_val, new_var);
                            match &mut self.instr[i as usize] {
                                TacOp::LoadImm(v, _) | TacOp::AssignOp(v, _, _, _) => {
                                    *v = new_var;
                                },
                                _ => unreachable!(),
                            }
                        } else {
                            var_map.output_vars.insert(*v, *v);
                        }
                    },
                    _ => {},
                }
            }
        }

        // --- Definition expansion and conflict management ---
        // Now that we know what each block defines, we should compute what expressions each block can
        // see, that is different from the one before because each block carries the definitions of
        // its predecessors, we also can't do a direct top to bottom single pass because of loops
        // So what to do? for each block merge it's output_vars with it's predecessors, particularly:
        // - If multiple predecessors offer different definition of the same variable, then create
        //   a conflict (that will be transcribed as a phi function).
        // - If a variable carried from a predecessor is redefined in the current block,
        //   the redefinition takes precedence
        // - If a variable carried from a predecessor is not redefined, it is carried over

        let mut changed = true;
        while changed {
            changed = false;
            for cfgi in 0..self.cfg.len() {
                let mut conflicts = HashMap::new();
                mem::swap(&mut conflicts, &mut cfg_vars[cfgi].conflicts);

                let prev_len = self.cfg[cfgi].prev.len();
                for prev_vars_i in 0..prev_len {
                    let prev_vars_i = self.cfg[cfgi].prev[prev_vars_i];
                    let prev_vars = cfg_vars[prev_vars_i].output_vars.clone();
                    for (orig_var, new_var) in prev_vars {
                        let old_var = *cfg_vars[cfgi].output_vars.entry(orig_var).or_insert(new_var);

                        if old_var != new_var {
                            // Conflict: our version of the variable is different from what the previous
                            match conflicts.get_mut(&orig_var) {
                                None => {
                                    // Huh? no conflict? *takes out American flag* let's bring in some democracy!
                                    // Create a conflict between new_var and var_map[orig_var]
                                    let res_var = self.create_variable(0, self.vars[old_var as usize].length, Some(orig_var));
                                    cfg_vars[cfgi].output_vars.insert(orig_var, res_var);
                                    conflicts.insert(orig_var, [old_var, new_var].iter().copied().collect());
                                    changed = true;
                                },
                                Some(vars) => {
                                    // Conflict already present, add the new_var to the conflict.
                                    changed |= vars.insert(new_var);
                                },
                            }
                        } else {
                            changed |= cfg_vars[cfgi].output_vars.insert(orig_var, new_var) != Some(new_var);
                        }
                    }
                }
                cfg_vars[cfgi].conflicts = conflicts;
            }
        }

        // Encode the obtained information on the CFG node (it will be useful in the un-ssa process)
        for (cfgi, cfg_data) in (0..self.cfg.len()).zip(cfg_vars.iter()) {
            let mut phi_origin = HashMap::new();
            for prev in self.cfg[cfgi].prev.iter().copied() {
                for (&orig_var, &new_var) in cfg_vars[prev].output_vars.iter() {
                    if cfg_data.input_vars.get(&orig_var).copied() != Some(new_var) {
                        phi_origin.entry(orig_var).or_insert(Vec::new()).push((new_var, prev as u32));
                    }
                }
            }
            self.cfg[cfgi].ssa_phi_origin = phi_origin;
        }


        for cfgi in 0..self.cfg.len() {
            let mut var_map = HashMap::new();

            // Create phi functions for every conflict
            for (orig_var, vars) in cfg_vars[cfgi].conflicts.iter() {
                let posi = self.cfg[cfgi].start;
                let vari = self.create_variable(posi, self.vars[*orig_var as usize].length, None);
                self.insert_instruction(posi, TacOp::PhiFun(vari, vars.iter().copied().collect()));
                var_map.insert(*orig_var, vari);
            }


            let start = self.cfg[cfgi].start;
            let end = self.cfg[cfgi].end;

            for i in start..end {
                let op = &mut self.instr[i as usize];
                match op {
                    TacOp::AssignOp(res, a, _, b) => {
                        apply_rename_map(&var_map, a);

                        match b {
                            Operator::Var(mut b) => {
                                apply_rename_map(&var_map, &mut b);
                            },
                            Operator::Imm(_) => {},
                        }
                    },
                    TacOp::BranchIfZero(_, v) | TacOp::BindArg(_, v) | TacOp::BindRet(_, v)  => {
                        apply_rename_map(&var_map, v);
                    },
                    _ => {},
                }
            }
        }
    }

    pub fn convert_out_of_ssa(&mut self) {
        for cfgi in 0..self.cfg.len() {
            let mut phi_origin = HashMap::new();
            mem::swap(&mut phi_origin, &mut self.cfg[cfgi].ssa_phi_origin);
            for (orig_var, sources) in phi_origin.into_iter() {
                for (var_id, cfg_src) in sources {
                    let end = self.cfg[cfg_src as usize].end;
                    self.insert_instruction(end, TacOp::AssignOp(orig_var, var_id, Operation::Add, Operator::Imm(0)));
                }
            }

            let start = self.cfg[cfgi].start;
            while let Some(TacOp::PhiFun(_, _)) = self.instr.get(start as usize) {
                self.erase_instruction(start);
            }
        }
    }
}