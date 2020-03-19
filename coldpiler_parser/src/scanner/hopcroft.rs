use std::collections::HashMap;
use std::hash::Hash;

use crate::util::Partition;
use crate::util::HashableSet;

use super::scanner::{CustomTokenType, Scanner};
use std::fmt::{Display, Formatter, Error, Write};
use std::cmp::Ordering;

#[derive(Hash, Debug, Clone, PartialEq, Eq)]
struct FullPartition(HashableSet<Partition>);


impl Display for FullPartition {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.write_char('{')?;
        for (index, partition) in (self.0).0.iter().enumerate() {
            if index != 0 {
                f.write_char(',')?;
            }
            write!(f, "{}", partition)?;
        }
        f.write_char('}')?;
        Ok(())
    }
}

struct HopcroftData<'a, T> where T: CustomTokenType + Hash {
    graph: &'a Scanner<T>,
    node_to_partition: HashMap<u32, FullPartition>,
}

impl<'a, T> HopcroftData<'a, T>  where T: CustomTokenType + Hash {
    fn new(original: &'a Scanner<T>) -> Self {
        HopcroftData {
            graph: original,
            node_to_partition: HashMap::new(),
        }
    }

    fn partition_by_token(&mut self) -> FullPartition {
        let mut part = HashMap::new();

        for (index, token) in self.graph.nodes().enumerate() {
            part.entry(*token).or_insert_with(Partition::create_empty).insert(index as u32);
        }

        FullPartition (
            part.drain().map(|e| { e.1 }).collect()
        )
    }

    fn split_partition(&self, partition: &Partition) -> FullPartition {
        if partition.len() == 1 {
            return FullPartition([partition].iter().map(|x| (*x).clone()).collect())
        };
        let mut splitted = HashMap::new();

        for &node in partition.iter() {
            let mut partitions = HashableSet::new();

            for char in 0..255u8 {
                if let Some(next_node) = self.graph.get_next(node, char) {
                    if !partition.contains(next_node) {
                        let x = self.node_to_partition.get(&next_node).unwrap();
                        partitions.insert((char, x));
                    }
                }
            }
            splitted.entry(partitions)
                .or_insert_with(Vec::new)
                .push(node)
        }
        FullPartition(
            splitted.drain().map(|mut x| {
                Partition::create_from(x.1.drain(..).collect())
            }).collect()
        )
    }

    fn update_node_to_partition(&mut self, fp: &FullPartition) {
        for x in fp.0.iter() {
            for &x in x.iter() {
                self.node_to_partition.insert(x, fp.clone());
            }
        }
    }
}

impl<T: CustomTokenType + Hash> Scanner<T> {
    pub fn minimize_hopcroft(&self) -> Scanner<T> {
        let mut data = HopcroftData::new(self);

        let mut todo_partition = data.partition_by_token();
        let mut partition = FullPartition(HashableSet::new());

        data.update_node_to_partition(&todo_partition);

        // println!("Hopcroft start: {}", self.to_nfa());

        while todo_partition.0.len() != partition.0.len() {
            // println!("Step: {} {}", todo_partition, partition);
            partition = todo_partition;
            todo_partition = FullPartition(HashableSet::new());

            for part in partition.0.iter() {
                let splitted = data.split_partition(part);
                data.update_node_to_partition(&splitted);
                todo_partition.0.union(&splitted.0);
            }
        }

        let mut res = Scanner::new(partition.0.len() as u32);

        let mut node_to_part_index = HashMap::new();

        let first_index = partition.0.iter().position(|p| {
            p.iter().any(|&x| x == 0)
        }).unwrap();

        let convert_pointer = |i: usize| {
            (match i.cmp(&first_index) {
                Ordering::Less => i + 1,
                Ordering::Equal => 0,
                Ordering::Greater => i,
            }) as u32
        };

        for (index, part) in partition.0.iter().enumerate() {
            let token = self.get_token(*part.iter().next().unwrap());

            let real_index = convert_pointer(index);
            // println!("PART {} to {}", part, real_index);

            if let Some(real_token) = token {
                res.set_token(real_index, real_token);
            }
            for &node in part.iter() {
                node_to_part_index.insert(node, real_index);
            }
        }

        for (index, part) in partition.0.iter().enumerate() {
            let real_index = convert_pointer(index);
            for node in part.iter() {
                for char in 0..255u8 {
                    if let Some(next_node) = self.get_next(*node, char) {
                        res.add_edge(real_index, node_to_part_index[&next_node], char);
                    }
                }
            }
        }

        res
    }
}


