use crate::tac::TacFunction;
use std::collections::{HashSet, VecDeque};

impl TacFunction {

    pub fn compute_height(&mut self) {
        let cfg = &mut self.cfg;

        cfg[0].height = 0;
        let mut queue = VecDeque::new();
        queue.push_back(0usize);

        while let Some(cfgi) = queue.pop_front() {
            let h = cfg[cfgi].height + 1;
            for x in cfg[cfgi].dominates.clone() {// TODO: remove clone
                if cfg[x].height > h {
                    cfg[x].height = h;
                    queue.push_back(x);
                }
            }
        }
    }

    pub fn compute_domination(&mut self) {
        // TODO: use bitset?
        let mut dom = vec![];
        dom.push(HashSet::new());
        dom[0].insert(0usize);
        let set: HashSet<_> = (1..self.cfg.len()).collect();
        dom.resize(self.cfg.len(), set);

        let mut changed = true;
        while changed {
            changed = false;
            for i in 1..self.cfg.len() {
                let mut iter = self.cfg[i].prev.iter().map(|x| &dom[*x]);
                let tmp = iter.next().expect("Unreachable CFG").clone();
                let tmp = iter.fold(tmp, |acc, item| {
                    acc.intersection(&item).copied().collect()
                });
                if tmp != dom[i] {
                    changed = true;
                    dom[i] = tmp;
                }
            }
        }

        for (i, d) in dom.iter().enumerate() {
            self.cfg[i].is_dominated = d.iter().copied().collect();
            for x in d.iter().copied() {
                self.cfg[x].dominates.push(i);
            }
        }

        self.compute_height();

        // Compute immediate dominators in O(n^2)
        for i in 0..self.cfg.len() {
            self.cfg[i].immediate_dominator = 0;
            let mut idom_h = 0;
            let dom_len = self.cfg[i].is_dominated.len();
            for j in 0..dom_len {
                let dom_i = self.cfg[i].is_dominated[j];
                if dom_i == i {
                    continue;
                }
                if self.cfg[dom_i].height > idom_h {
                    idom_h = self.cfg[j].height;
                    self.cfg[i].immediate_dominator = j;
                }
            }
        }
    }
}

