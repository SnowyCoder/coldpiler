use crate::parser::grammar::{GrammarToken, GrammarTokenType};
use crate::scanner::CustomTokenType;

pub type NodeIndex = usize;

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SyntaxTree<T: CustomTokenType, N: GrammarTokenType> {
    nodes: Vec<SyntaxNode<T, N>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyntaxNode<T: CustomTokenType, N: GrammarTokenType> {
    pub parent: Option<NodeIndex>,
    pub gtype: GrammarToken<T, N>,
    pub text: Option<String>,
    pub children: Vec<NodeIndex>,
}

impl<T: CustomTokenType, N: GrammarTokenType> SyntaxNode<T, N> {
    pub fn new(gtype: GrammarToken<T, N>, text: Option<String>) -> Self {
        SyntaxNode {
            parent: None,
            gtype,
            text,
            children: Vec::new(),
        }
    }
}

impl<T: CustomTokenType, N: GrammarTokenType> SyntaxTree<T, N> {
    pub fn new() -> Self {
        SyntaxTree {
            nodes: Vec::new()
        }
    }

    pub fn from_nodes_unchecked(nodes: Vec<SyntaxNode<T, N>>) -> Self {
        SyntaxTree {
            nodes
        }
    }

    pub fn node_exists(&self, node: NodeIndex) -> bool {
        node < self.nodes.len()
    }

    pub fn create_node(&mut self, gtype: GrammarToken<T, N>, text: Option<String>, parent: Option<NodeIndex>) -> NodeIndex {
        let mut node = SyntaxNode::new(gtype, text);
        let node_index = self.nodes.len();
        if let Some(par) = parent {
            let parent_node = self.nodes.get_mut(par).expect("Parent not found");
            node.parent = parent;
            parent_node.children.push(node_index);
        }
        self.nodes.push(node);
        node_index
    }

    pub fn reassign_parent(&mut self, node: NodeIndex, new_parent: Option<NodeIndex>) {
        assert!(new_parent.map_or(true, |x| self.node_exists(x)));

        let real_node = self.nodes.get_mut(node).expect("Invalid node index");

        let old_parent = real_node.parent;
        real_node.parent = new_parent;

        if let Some(current_parent) = old_parent {
            let childrens = &mut self.nodes[current_parent].children;
            let index_in_parent = childrens.iter().position(|&x| x == node).unwrap();
            childrens.remove(index_in_parent);
        }

        if let Some(par) = new_parent {
            self.nodes[par].children.push(node);
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn node(&self, index: NodeIndex) -> &SyntaxNode<T, N> {
        &self.nodes[index]
    }

    pub fn mut_node(&mut self, index: NodeIndex) -> &mut SyntaxNode<T, N> {
        &mut self.nodes[index]
    }

    pub fn find_root(&self) -> Option<NodeIndex> {
        // Usually the root is the first or the last element (optimization)
        // So check the first, then start from the right and search to the left.
        if let Some(first) = self.nodes.first() {
            if first.parent == None {
                return Some(0)
            }
        } else {
            return None
        }
        self.nodes.iter()
            .position(|x| x.parent == None)
    }
}
