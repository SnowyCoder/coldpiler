use std::collections::{HashMap, BTreeSet, BTreeMap};
use std::hash::Hash;

use crate::util::Partition;

use super::scanner::{CustomTokenType, Scanner};
use std::fmt::{Display, Formatter, Error, Write, Debug};
use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd)]
struct FullPartition(BTreeSet<Partition>);


impl Display for FullPartition {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.write_char('{')?;
        for (index, partition) in (self.0).iter().enumerate() {
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
    partitions: PartitionRegistry,
}

/// Partitions are BTree<u32> so they're quite heavy objects to manage and compare,
/// to work around that this registry registers every partition present and it assigns an
/// incrementing id to each one.
/// It also manages the node to partition search.
/// TODO: this data structure contains some kind of safe "memory-leak", meaning that even when
/// the partition is not used anymore it will still be registered in the registry.
/// This might not be a problem since the algorithm doesn't create that many partitions.
struct PartitionRegistry {
    id_to_partition: Vec<Partition>,
    node_to_partition_id: Vec<u32>,
}

impl PartitionRegistry {
    fn from_part(node_count: usize, initial_part: &FullPartition) -> Self {
        let mut id_to_partition = Vec::with_capacity(initial_part.0.len());
        let mut node_to_partition_id = vec![0; node_count];

        for partition in initial_part.0.iter() {
            let id = id_to_partition.len() as u32;
            id_to_partition.push(partition.clone());

            for node in partition.iter() {
                node_to_partition_id[*node as usize] = id;
            }
        }

        PartitionRegistry {
            id_to_partition,
            node_to_partition_id
        }
    }

    fn search_node_partition(&self, node: u32) -> u32 {
        *self.node_to_partition_id.get(node as usize)
            .expect("Cannot find node in partition registry")
    }

    fn insert_part(&mut self, part: Partition) -> u32 {
        if let Some(old_id) = self.id_to_partition.iter().position(|x| *x == part) {
            return old_id as u32;
        }
        let id = self.id_to_partition.len() as u32;
        // Update node_to_partition
        for node in part.iter() {
            self.node_to_partition_id[*node as usize] = id;
        }
        self.id_to_partition.push(part);

        id
    }

    fn insert_full_part(&mut self, part: &FullPartition) {
        for part in part.0.iter() {
            self.insert_part(part.clone());
        }
    }
}

/// Summarize the behaviour of a state, this is explained with more depth in the split_partition
/// method within HopcroftData.
/// Basically it's an implementation of Ord and Eq for an array of 256 bits
#[derive(Clone)]
struct StateBehaviour([u32; 256]);

impl PartialOrd for StateBehaviour {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for StateBehaviour {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialEq for StateBehaviour {
    fn eq(&self, other: &Self) -> bool {
        let s: &[u32] = &self.0;
        let o: &[u32] = &other.0;
        s.eq(o)
    }
}

impl Eq for StateBehaviour {
}

impl<'a, T> HopcroftData<'a, T>  where T: CustomTokenType + Hash {
    fn new(original: &'a Scanner<T>, partitions: &FullPartition) -> Self {
        HopcroftData {
            graph: original,
            partitions: PartitionRegistry::from_part(original.get_node_count() as usize, partitions),
        }
    }

    fn partition_by_token(scanner: &Scanner<T>) -> FullPartition {
        let mut part = HashMap::new();

        for (index, token) in scanner.nodes().enumerate() {
            part.entry(*token).or_insert_with(Partition::create_empty).insert(index as u32);
        }

        FullPartition (
            part.drain().map(|e| { e.1 }).collect()
        )
    }

    fn get_behaviour(&self, state: u32) -> StateBehaviour {
        let null: u32 = u32::max_value();
        let mut res = [null; 256];
        for char in 0..255u8 {
            let next_part = self.graph.get_next(state, char)
                .map(|to_state| self.partitions.search_node_partition(to_state));
            if let Some(out_state) = next_part {
                res[char as usize] = out_state;
            }
        }
        StateBehaviour(res)
    }

    fn split_partition(&self, partition: &Partition) -> FullPartition {
        // What does it mean to split a partition?
        // Every node in a partition should have the same behaviour, that means that
        // for every character in the alphabet the nodes should either jump to the same partition or
        // reject the input (all at the same time).
        // Every jump inside the partition is useless.
        // Splitting a partition means dividing the nodes based on their behaviour and separating them
        // in more partitions.
        // To do that we divide the various nodes in the partition by their behaviour, that means
        // that we extract the destination partition from every node and so we repartition the nodes
        // based on that.

        // A partition with just one node is already refined.
        if partition.len() == 1 {
            return FullPartition([partition].iter().map(|x| (*x).clone()).collect())
        };
        let mut nodes_by_behaviour = BTreeMap::new();

        for node in partition.iter() {
            let behaviour = self.get_behaviour(*node);

            nodes_by_behaviour.entry(behaviour)
                .or_insert_with(Vec::new)
                .push(*node);
        }

        FullPartition(
            nodes_by_behaviour.iter().map(|(_behaviour, nodes)| {
                Partition::create_from(nodes.iter().cloned().collect())
            }).collect()
        )
    }
}

impl<T: CustomTokenType + Hash> Scanner<T> {
    pub fn minimize_hopcroft(&self) -> Scanner<T> {
        let mut todo_partition = HopcroftData::partition_by_token(self);
        let mut partition = FullPartition(BTreeSet::new());

        let mut data = HopcroftData::new(self, &todo_partition);

        // The core of the algorithm:
        // Now we use two partitions: the current one and the next one
        // We try to split every partition until we can't split anymore,
        // The "split" procedure is explained in HopcroftData::split
        // The idea here is to start with the best optimization ever (that is, one partition for
        // each token type, with some edges in between), then we refine these partitions based on
        // the nodes behaviour (check the split_partition method for more details on this).
        while todo_partition.0.len() != partition.0.len() {
            partition = todo_partition;
            todo_partition = FullPartition(BTreeSet::new());

            for part in partition.0.iter() {
                let mut splitted = data.split_partition(part);
                data.partitions.insert_full_part(&splitted);
                todo_partition.0.append(&mut splitted.0);
            }
        }

        // When we're done each partition should represent a group of nodes with the same behaviour.
        // This means that we can now merge all the nodes in that partition in a single node with
        // the same connections.
        // Or, to view it in another way, the partitions are the nodes!
        let mut res = Scanner::new(partition.0.len() as u32);

        let mut node_to_part_index = HashMap::new();

        // Problem: 0 is the starting point in the DFA and this shouldn't change
        // That means that when we assign the indexes the partition with the 0 node should stay 0.
        // So here we find the index of the 0 partition
        let first_index = partition.0.iter().position(|p| {
            p.iter().any(|&x| x == 0)
        }).unwrap();

        // Here we convert the partition indexes to the real index, that we will use in the DFA.
        // Basically we shift any node before the 0 partition right (to make space) and we put the
        // 0 partition in the 0 index.
        let convert_pointer = |i: usize| {
            (match i.cmp(&first_index) {
                Ordering::Less => i + 1,
                Ordering::Equal => 0,
                Ordering::Greater => i,
            }) as u32
        };

        for (index, part) in partition.0.iter().enumerate() {
            // The token for any node should suffice, we initially divided the partitions based on
            // the nodes token so any node in the partition will have the same token.
            let token = self.get_token(*part.iter().next().unwrap());

            let real_index = convert_pointer(index);

            if let Some(real_token) = token {
                res.set_token(real_index, real_token);
            }
            for &node in part.iter() {
                node_to_part_index.insert(node, real_index);
            }
        }

        // Now that we have filled the node_to_part_index map we can fill out the edges.
        // But first some proofs: the add_edge procedure is not completely "safe": it will panic if
        // another node is already connected to that node with the same char, can we be sure that it
        // will not happend?
        // Theoretically: yes!
        // Every node in the partition has the same behaviour, so if a node 0 points to partition 1
        // with the key 'a', then node 1 should also point to some node of partition 1 with the
        // key 'a'.
        // This means that every node on the partition will generate the same partition-to-partition
        // edges, so we can only use the first node in the partition to generate them as the others
        // will just be duplicated.
        // (this seems quite stupid now, but it has generated a bug in the first commit when the
        // split_partition implementation was bugged, and it has caused me some panic nightmares)

        for (index, part) in partition.0.iter().enumerate() {
            let real_index = convert_pointer(index);
            let node = part.iter().next().unwrap();
            for char in 0..255u8 {
                if let Some(next_node) = self.get_next(*node, char) {
                    res.add_edge(real_index, node_to_part_index[&next_node], char);
                }
            }
        }

        res
    }
}


