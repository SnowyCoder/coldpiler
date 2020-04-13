use crate::parser::grammar::{GrammarToken, GrammarTokenType};
use crate::scanner::{ScannerTokenType, TokenLoc};
use crate::loc::SpanLoc;

pub type NodeIndex = usize;

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SyntaxTree<T: ScannerTokenType, N: GrammarTokenType> {
    nodes: Vec<SyntaxNode<T, N>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyntaxNode<T: ScannerTokenType, N: GrammarTokenType> {
    pub parent: Option<NodeIndex>,
    pub gtype: GrammarToken<T, N>,
    pub text: Option<TokenLoc>,
    pub span: SpanLoc,
    pub children: Vec<NodeIndex>,
}

impl<T: ScannerTokenType, N: GrammarTokenType> SyntaxNode<T, N> {
    pub fn new(gtype: GrammarToken<T, N>, span: SpanLoc, text: Option<TokenLoc>) -> Self {
        SyntaxNode {
            parent: None,
            gtype,
            text,
            span,
            children: Vec::new(),
        }
    }
}

impl<T: ScannerTokenType, N: GrammarTokenType> SyntaxTree<T, N> {
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

    pub fn create_node(&mut self, gtype: GrammarToken<T, N>, span: SpanLoc, text: Option<TokenLoc>, parent: Option<NodeIndex>) -> NodeIndex {
        let mut node = SyntaxNode::new(gtype, span, text);
        let node_index = self.nodes.len();
        if let Some(par) = parent {
            let parent_node = self.nodes.get_mut(par).expect("Parent not found");
            node.parent = parent;
            parent_node.children.push(node_index);
        }
        self.nodes.push(node);
        node_index
    }

    fn expand_span(&mut self, mut node: NodeIndex, mut other_span: SpanLoc) {
        loop {
            let real_node = self.nodes.get_mut(node).unwrap();

            let prev_span = real_node.span;

            real_node.span.merge(other_span);

            if prev_span == real_node.span {
                break
            }
            if let Some(par) = real_node.parent {
                node = par;
                other_span = real_node.span;
            } else {
                break
            }
        }
    }

    pub fn reassign_parent(&mut self, node: NodeIndex, new_parent: Option<NodeIndex>) {
        assert!(new_parent.map_or(true, |x| self.node_exists(x)));

        let real_node = self.nodes.get_mut(node).expect("Invalid node index");

        let old_parent = real_node.parent;
        real_node.parent = new_parent;
        let span = real_node.span;

        if let Some(current_parent) = old_parent {
            let childrens = &mut self.nodes[current_parent].children;
            let index_in_parent = childrens.iter().position(|&x| x == node).unwrap();
            childrens.remove(index_in_parent);
        }

        if let Some(par) = new_parent {
            self.nodes[par].children.push(node);
            self.expand_span(par, span);
        }
    }

    // Returns the span of the last terminal node
    fn recompute_spans(&mut self, mut last_terminal: SpanLoc, node: NodeIndex) -> SpanLoc {
        let real_node = self.nodes.get_mut(node).unwrap();

        if let GrammarToken::Terminal(_) = real_node.gtype {
            return real_node.span;
        }
        // TODO: remove clone
        // Come oooon ruuuuust
        for child in self.nodes[node].children.clone() {
            last_terminal = self.recompute_spans(last_terminal, child);
        }
        let span = self.nodes[node].children.iter()
            .map(|x| self.nodes[*x].span)
            .fold(None, |acc: Option<SpanLoc>, curr| {
                if let Some(mut acc) = acc {
                    acc.merge(curr);
                    Some(acc)
                } else {
                    Some(curr)
                }
            });
        self.nodes.get_mut(node).unwrap().span = span.unwrap_or(last_terminal);

        last_terminal
    }

    pub fn recompute_all_spans(&mut self, root: NodeIndex) {
        self.recompute_spans(SpanLoc::zero(), root);
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
