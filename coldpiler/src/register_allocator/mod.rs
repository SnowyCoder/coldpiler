use crate::tac::{TacProgram, TacFunction, VarId, CodeLoc, TacOp, Operator};
use std::collections::{BTreeMap, HashMap};
use std::intrinsics::transmute;
use std::cmp::Ordering;
use std::env::var;

// TODO: we could create a SSA register allocator as shown here: http://compilers.cs.ucla.edu/fernando/projects/soc/reports/short_tech.pdf

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum SavedBy {
    Caller, Callee
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum ConvRegType {
    None, Argument,
}

pub struct SingleRegisterConfig {
    pub internal_id: u16,
    pub saved_by: SavedBy,
    pub ctype: ConvRegType,
}

impl SingleRegisterConfig {
    pub fn of(internal_id: u16, saved_by: SavedBy, ctype: ConvRegType) -> Self {
        SingleRegisterConfig {
            internal_id,
            saved_by,
            ctype
        }
    }
}

pub struct RegisterConfig {
    pub regs: Vec<SingleRegisterConfig>,
    pub static_zero: Option<u16>,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum VariableStorage {
    Register(u16),
    Spill(VarId),
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum VarAllocAction {
    //       var   reg    spill load/store
    Alloc  (VarId, u16, Option<u16>),
    DeAlloc(VarId, u16, Option<u16>),
}

pub struct VariablesDisposition {
    spill_actions: Vec<(CodeLoc, VarAllocAction)>,
    pub concurr_spill_count: u32,
    code_loc: u32,
    spill_index: u32,
}

impl VariablesDisposition {
    pub fn advance_code(&mut self) -> impl Iterator<Item=VarAllocAction> + '_ {
        loop {
            let curr_loc = match self.spill_actions.get(self.spill_index as usize) {
                None => break,
                Some(x) => x.0,
            };
            if curr_loc >= self.code_loc {
                break
            }
            self.spill_index += 1;
        }

        let (spill_begin, spill_end) =
            if self.spill_actions.get(self.spill_index as usize).map_or(false, |x| x.0 != self.code_loc) {
                (0, 0)
            } else {
                let spill_begin = self.spill_index as usize;
                loop {
                    let curr_loc = match self.spill_actions.get(self.spill_index as usize) {
                        None => break,
                        Some(x) => x.0,
                    };
                    if curr_loc != self.code_loc {
                        break
                    }
                    self.spill_index += 1;
                }
                let spill_end = self.spill_index as usize;
                (spill_begin, spill_end)
            };

        self.code_loc += 1;

        self.spill_actions.iter()
            .skip(spill_begin)
            .take(spill_end - spill_begin)
            .map(|x| x.1)
    }
}

impl RegisterConfig {
    pub fn allocate(&self, cnf: &TacFunction) -> VariablesDisposition {
        let mut fun = cnf.clone();
        let mut data = vec![VirtRegData::new(); fun.vars.len()];

        let mut translation_map = HashMap::new();

        loop {
            fun.recompute_live_ranges();
            construct_regdata(&fun, &mut data);

            match solve_graph(self, &mut data, vec![]) {
                GraphColoringResult::Success => break,
                GraphColoringResult::ErrorSpill(x) => {
                    data[x as usize].is_spilled = true;
                    eprintln!("Spilling {}", x);
                    spill_rewrite(&mut fun, x, x, &mut translation_map);
                },
                GraphColoringResult::ErrorUnsolvable => panic!("Cannot allocate registers"),
            }
        }

        /*eprintln!("Graph solved: {:?}", data);
        for (i, d) in data.iter().enumerate() {
            eprintln!("x{} -> {} {}", i, d.color, self.regs[d.color as usize].internal_id);
        }*/
        let ranges: Vec<_> = fun.vars.iter()
            .enumerate()
            .map(|(i, v)| (v.live_range, i as u32))
            .collect();
        let mut open_ranges = ranges.clone();
        let mut close_ranges = ranges;
        open_ranges.sort_unstable_by_key(|((from, to), i)| *from);
        close_ranges.sort_unstable_by_key(|((from, to), i)| *to);

        // Sort ranges in two vectors, the first one will be sorted by the range openings in
        // crescent order, the other will be sorted by the range closing in crescent order.
        // This allows a clear forward iteration when searching what variables will be allocated and
        // deallocated in a certain index.

        let mut actions = Vec::new();

        let mut origin_index = 0;
        let mut open_range_i = 0;
        let mut close_range_i = 1;

        for (instr_i, instr) in fun.instr.iter().enumerate() {
            while let Some(((f, t), virt)) = open_ranges.get(open_range_i){
                match f.cmp(&(instr_i as u32)) {
                    Ordering::Less => {},
                    Ordering::Equal => {
                        if fun.vars[*virt as usize].is_zero {
                            actions.push((origin_index, VarAllocAction::Alloc(*virt, self.static_zero.unwrap(), None)))
                        } else {
                            let orig = translation_map.get(virt).copied().unwrap_or(*virt);
                            let d = &data[*virt as usize];
                            if d.is_colored {
                                let reg = self.regs[d.color as usize].internal_id;
                                let spill = if d.is_spilled {
                                    Some(orig as u16)
                                } else {
                                    None
                                };
                                actions.push((origin_index, VarAllocAction::Alloc(orig, reg, spill)));
                            }
                        }
                    },
                    Ordering::Greater => {
                        break;
                    },
                }
                open_range_i += 1;
            }
            while let Some(((f, t), virt)) = close_ranges.get(close_range_i){
                match t.cmp(&(instr_i as u32)) {
                    Ordering::Less => {},
                    Ordering::Equal => {
                        if fun.vars[*virt as usize].is_zero {
                            actions.push((origin_index, VarAllocAction::DeAlloc(*virt, self.static_zero.unwrap(), None)))
                        } else {
                            let orig = translation_map.get(virt).copied().unwrap_or(*virt);
                            let d = &data[*virt as usize];
                            if d.is_colored {
                                let reg = self.regs[d.color as usize].internal_id;
                                let spill = if d.is_spilled {
                                    Some(orig as u16)
                                } else {
                                    None
                                };
                                actions.push((origin_index, VarAllocAction::DeAlloc(orig, reg, spill)));
                            }
                        }
                    },
                    Ordering::Greater => {
                        break;
                    },
                }
                close_range_i += 1;
            }

            match instr {
                TacOp::Spill(_, _) | TacOp::UnSpill(_, _)=> {
                },
                _ => {
                    origin_index += 1;
                }
            }
        }

        VariablesDisposition {
            spill_actions: actions,
            concurr_spill_count: cnf.vars.len() as u32,
            code_loc: 0,
            spill_index: 0
        }
    }
}

pub struct RegisterAllocator {
    config: RegisterConfig,
}

#[derive(Clone, Debug)]
pub struct VirtRegData {
    pub color: u16,
    pub is_colored: bool,
    pub in_stack: bool,
    pub is_trouble: bool,
    pub is_spilled: bool,
    pub edges: Vec<u32>,
}

impl VirtRegData {
    pub fn new() -> Self {
        VirtRegData {
            color: 0,
            is_colored: false,
            in_stack: false,
            is_trouble: false,
            is_spilled: false,
            edges: vec![],
        }
    }
}

pub enum GraphColoringResult {
    Success,
    ErrorSpill(u32),
    // what if we try to spill a spilled variable? well, this happens
    // And I can't prove that it cannot happen, so there's that.
    // Example:
    // A = 1;
    // B = 2;
    // C = 3;
    // D = A + B;
    // E = D + C;
    // In this program the graph will be
    //   A
    //  / \
    // B - C - D   E
    // And we won't be able to color it with a 2 register machine
    // If we try to spill C we'll have:
    // A = 1;
    // B = 2;
    // C = 3;
    // spill(C);
    // D = A + B;
    // C1 = reload(C);
    // E = D + C1;
    // Or, using graphs:
    //   A
    //  / \
    // B - C
    // C1 - D
    // E
    // The A-B-C circle is still there so the program might still want to spill C.
    // We could try to not re-spill a variable and chose another one but it is not a sound algorithm
    // for me, and even if it works (ir probably would but I can't prove it), it's quite ugly.
    ErrorUnsolvable,
}

fn construct_regdata(fun: &TacFunction, data: &mut Vec<VirtRegData>) {
    for x in data.iter_mut() {
        x.in_stack = false;
        x.is_trouble = false;
        x.is_colored = false;
        x.edges.clear();
    }

    let mut ranges: Vec<_> = fun.vars.iter()
        .enumerate()
        .map(|(i, v)| {
            (v.live_range, i)
        }).collect();

    ranges.sort_unstable();
    //eprintln!("Ranges: {:?}", ranges);

    for (index, ((from, to), var_index)) in ranges.iter().copied().enumerate() {
        if fun.vars[var_index].is_zero {
            continue;
        }
        let mut edges: Vec<_> = ranges.iter()
            .skip(index + 1)
            .filter(|((_, _), vi)| !fun.vars[*vi].is_zero)
            .take_while(|((f, _), _)| to > *f)
            .map(|((_, _), var)| *var as u32)
            .collect();
        for x in edges.iter() {
            data[*x as usize].edges.push(var_index as u32);
        }
        data[var_index as usize].edges.append(&mut edges);
    }
    /*for (i, x) in data.iter().enumerate() {
        eprintln!(" {} edges: {:?}", i, x.edges);
    }*/
}

fn solve_graph(reg: &RegisterConfig, data: &mut Vec<VirtRegData>, coalesce: Vec<(u32, u32)>) -> GraphColoringResult {
    let mut stack = Vec::new();

    let regs_len = reg.regs.len();

    loop {
        let mut colourable_i = -1;
        let mut safely_colourable_i = -1;
        for (index, virt) in data.iter().enumerate() {
            if virt.is_colored || virt.in_stack {
                continue // Already colored or in stack
            }

            colourable_i = index as isize;
            let edges = virt.edges.iter()
                .cloned()
                .filter(|e| data[*e as usize].in_stack)
                .count();
            if edges < regs_len {
                safely_colourable_i = index as isize;
                break;
            }
        }
        if safely_colourable_i != -1 {
            data[safely_colourable_i as usize].in_stack = true;
            stack.push(safely_colourable_i as u32);
        } else if colourable_i != -1 {
            // There are some non-colored virts but they have more than k-1 edges,
            // something might spill!
            data[colourable_i as usize].is_trouble = true;
            data[colourable_i as usize].in_stack = true;
            stack.push(colourable_i as u32);
        } else {
            // No more node to stack, start colouring them
            break;
        }
    }

    while let Some(curr) = stack.pop() {
        let curri = curr as usize;
        data[curri].in_stack = false;

        let mut colours = 0u64;
        for edge_i in data[curri].edges.iter().copied() {
            let edge = &data[edge_i as usize];
            if edge.is_colored {
                colours |= (1 << edge.color as u64);
            }
        }

        let mut chosen_color = -1;
        for reg in 0..(regs_len as u64) {
            if (colours & (1 << reg)) == 0 {
                chosen_color = reg as i64;
                break;
            }
        }
        if chosen_color != -1 {
            data[curri].color = chosen_color as u16;
            data[curri].is_colored = true;
        } else {
            data[curri].is_colored = false;
            return GraphColoringResult::ErrorSpill(curri as u32);
        }
    }
    return GraphColoringResult::Success;
}

fn spill_rewrite(fun: &mut TacFunction, virt: u32, mem_pos: u32, translation_map: &mut HashMap<u32, u32>) {
    let mut i = 0u32;

    while let Some(_) = fun.instr.get(i as usize) {
        // Read
        let mut found = false;
        let new_var = fun.vars.len() as u32;
        match &mut fun.instr[i as usize] {
            TacOp::PhiFun(_, vars) => {
                for v in vars.iter_mut() {
                    if *v == virt {
                        *v = new_var;
                        found = true;
                    }
                }
            },
            TacOp::AssignOp(_, a, _, b) => {
                if *a == virt {
                    *a = new_var;
                    found = true;
                }
                if let Operator::Var(v) = b {
                    if *v == virt {
                        *v = new_var;
                        found = true;
                    }
                }
            },
            TacOp::BranchIfZero(_, v) | TacOp::BindArg(_, v) => {
                if *v == virt {
                    *v = new_var;
                    found = true;
                }
            },
            _ => {},
        }
        if found {
            let var = fun.create_variable(i, fun.vars[virt as usize].length, None);
            // Same as new_var
            fun.insert_instruction(i, TacOp::UnSpill(var, mem_pos));
            translation_map.insert(virt, new_var);
        }

        // Write
        match &mut fun.instr[i as usize] {
            TacOp::LoadImm(v, _) | TacOp::PhiFun(v, _) | TacOp::AssignOp(v, _, _, _) | TacOp::BindRet(_, v) => {
                if *v == virt {
                    let new_var = fun.vars.len() as u32;
                    *v = new_var;
                    let var = fun.create_variable(i, fun.vars[new_var as usize].length, None);
                    // Same as *v
                    fun.insert_instruction(i + 1, TacOp::Spill(var, mem_pos));
                    i += 1;
                }
            },
            _ => {},
        }
    }

}



