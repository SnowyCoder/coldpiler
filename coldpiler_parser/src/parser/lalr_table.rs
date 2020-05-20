use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::fmt;
use std::hash::Hash;

use coldpiler_util::Enumerable;

use crate::parser::{Grammar, GrammarToken, GrammarTokenType, SyntaxTree, ParsingError};
use crate::parser::grammar::GrammarRuleIndex;
use crate::scanner::{ScannerTokenType, Token};
use crate::util::{index_twice, IndexTwice};
use crate::loc::SpanLoc;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Action {
    Shift(u32),
    Reduce(u32),
    Accept,
    Reject,
}

pub struct ShiftReducer<T: ScannerTokenType + Enumerable, N: GrammarTokenType> {
    root_node_type: GrammarToken<T, N>,
    state_count: u32,
    action_table: Vec<Action>,
    goto_table: Vec<u32>,
    rules: Vec<(N, u32)>,
    ignore_tokens: Vec<T>,
}

impl<T: ScannerTokenType + Enumerable + 'static, N: GrammarTokenType + 'static> Display for ShiftReducer<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Root: {:?}", self.root_node_type)?;
        write!(f, "  ")?;
        for x in T::enumerate() {
            write!(f, "|{:10}", format!("{:?}", x))?;
        }
        write!(f, "|{:10}", "-")?;
        for x in N::enumerate() {
            write!(f, "|{:10}", format!("{:?}", x))?;
        }
        writeln!(f)?;
        for y in 0..self.state_count {
            write!(f, "{:<2}", y)?;
            for x in T::enumerate() {
                let a = self.get_action(y, Some(x));
                write!(f, "|{:10}", format!("{:?}", a))?;
            }
            write!(f, "|{:10}", format!("{:?}", self.get_action(y, None)))?;
            for token in N::enumerate() {
                let a = self.get_goto_raw(y, token);
                let a = match a {
                    None => "-".to_string(),
                    Some(x) => format!("{}", x),
                };

                write!(f, "|{:10}", a)?;
            }

            writeln!(f)?;
        }
        writeln!(f, "Rules:")?;
        for (index, (token, len)) in self.rules.iter().enumerate() {
            writeln!(f, "{}> {:?} {:?}", index, token, len)?;
        }
        writeln!(f, "Ignore tokens:")?;
        for token in self.ignore_tokens.iter() {
            writeln!(f, "- {:?}", token)?;
        }
        Ok(())
    }
}

impl<T: ScannerTokenType + Enumerable + 'static, N: GrammarTokenType + 'static> ShiftReducer<T, N> {
    pub fn from_raw(
        root_node_type: GrammarToken<T, N>,
        state_count: u32,
        action_table: Vec<Action>,
        goto_table: Vec<u32>,
        rules: Vec<(N, u32)>,
        ignore_tokens: Vec<T>
    ) -> Self {
        ShiftReducer { root_node_type, state_count, action_table, goto_table, rules, ignore_tokens }
    }

    #[cfg(feature = "codegen")]
    pub fn to_raw_code(&self, terminal_names: &[&proc_macro2::Ident], nonterminal_names: &[&proc_macro2::Ident]) -> proc_macro2::TokenStream {
        use quote::quote;

        let root_node = match self.root_node_type {
            GrammarToken::Terminal(x) => {
                let name = terminal_names[x.index()];
                quote! { GrammarToken::Terminal(T::#name) }
            },
            GrammarToken::NonTerminal(y) => {
                let name = nonterminal_names[y.index()];
                quote! { GrammarToken::NonTerminal(N::#name) }
            },
        };
        let state_count = self.state_count;

        let action_table = self.action_table.iter()
            .map(|a| {
            match a {
                Action::Shift(x) => quote! { Action::Shift(#x) },
                Action::Reduce(x) => quote! { Action::Reduce(#x) },
                Action::Accept => quote! { Action::Accept },
                Action::Reject => quote! { Action::Reject },
            }
        });

        let goto_table = self.goto_table.iter();

        let rules = self.rules.iter().map(|r| {
            let tokeni = r.0.index();
            let token_name = nonterminal_names[tokeni];
            let elemc = r.1;
            quote! {
                (N::#token_name, #elemc)
            }
        });

        let ignore_tokens = self.ignore_tokens.iter().map(|x| {
            let name = terminal_names[x.index()];
            quote! {
                T::#name
            }
        });

        quote! {
            #root_node,
            #state_count,
            vec![#(#action_table), *],
            vec![#(#goto_table), *],
            vec![#(#rules), *],
            vec![#(#ignore_tokens), *]
        }
    }

    pub fn action_index(state: u32, token: Option<T>) -> usize {
        let token_len = T::enumerate().count() as u32;
        let token_index = token.map_or(token_len, |x| x.index() as u32);
        let real_index = state * (token_len + 1) + token_index;
        real_index as usize
    }

    pub fn get_action(&self, state: u32, token: Option<T>) -> Action {
        self.action_table[Self::action_index(state, token)]
    }

    pub fn set_action(&mut self, state: u32, token: Option<T>, value: Action) -> Result<(), LRConflict<T, N>> {
        let index = Self::action_index(state, token);

        let old_val = self.action_table[index];
        if old_val != Action::Reject && old_val != value {
            // TODO: can we detect and display ambiguities in the language?
            // This IS a conflict, but why is there a conflict:
            // The lookahead has already been controlled so that means that the language is invalid?
            // It might be, but here's another case:
            // <Expr> := Number | <Expr> Plus <Expr>
            // This will generate a state in the DFA that's like this:
            // <Expr> := <Expr> * Plus <Expr> *
            // (where * is where's the reading point).
            // This means that the DFA will gladly shift a Plus, but will also reduce the Expr
            // and then use the rule <E> := <E> * P <E>
            // This means that there is a tree ambiguity:
            // To visualize this: the phrase:
            // Number Plus Number Plus Number
            // Can be constructed either by
            //       E
            //   E   | \
            // / | \ | |
            // N P N P N
            // or by
            //   E
            // / |   E
            // | | / | \
            // N P N P N
            // So try to disambiguate your grammar if you're sure it's LALR(1).
            return Err(LRConflict::Action(state, token, old_val, value))
        }
        self.action_table[index] = value;
        Ok(())
    }

    pub fn goto_index(&self, state: u32, token: N) -> usize {
        (state * N::enumerate().len() as u32 + token.index() as u32) as usize
    }

    pub fn get_goto_raw(&self, state: u32, token: N) -> Option<u32> {
        let res = self.goto_table[self.goto_index(state, token)];
        if res == u32::max_value() {
            None
        } else {
            Some(res)
        }
    }

    pub fn get_goto(&self, state: u32, token: N) -> u32 {
        match self.get_goto_raw(state, token) {
            None => {
                panic!(format!("Called get_goto on invalid input: state {}, token {:?}", state, token))
            },
            Some(x) => x,
        }
    }

    pub fn set_goto(&mut self, state: u32, token: N, dest: u32) -> Result<(), LRConflict<T, N>>{
        let index = self.goto_index(state, token);

        let old_val = self.goto_table[index];
        if old_val != u32::max_value() && old_val != dest {
            return Err(LRConflict::Goto(state, token, self.goto_table[index], dest));
            //panic!("Conflict in goto[{}][{:?}]:  {} or {}, is your grammar LR(1)?", state, token, self.goto_table[index], dest);
        }

        self.goto_table[index] = dest;
        Ok(())
    }

    fn advance_token(&self, tokens: &[Token<T>], mut next_index: usize) -> usize {
        while let Some(token) = tokens.get(next_index) {
            if !self.ignore_tokens.contains(&token.ttype) {
                break;
            }
            next_index += 1;
        }
        next_index
    }

    pub fn find_possible_tokens(&self, state: u32) -> Vec<Option<T>> {
        let mut res = Vec::new();
        for x in T::enumerate() {
            match self.get_action(state, Some(x)) {
                Action::Reject => {},
                _ => {
                    res.push(Some(x))
                }
            }
        }
        match self.get_action(state, None) {
            Action::Reject => {},
            _ => res.push(None)
        }
        res
    }

    pub fn parse(&self, tokens: &[Token<T>]) -> Result<SyntaxTree<T, N>, ParsingError<T>> {
        // This link: http://lambda.uta.edu/cse5317/notes/node18.html while short helped me
        // understand how and why this works
        let mut tree = SyntaxTree::new();

        let mut stack = vec![(0u32, usize::max_value())];

        let mut next_index = self.advance_token(tokens, 0);

        loop {
            let (top_state, _top_node) = *stack.last().unwrap();
            let lookahead = tokens.get(next_index);

            let lookahead_type =  lookahead.map(|x| x.ttype);
            let current_action = self.get_action(top_state, lookahead_type);
            //println!("{}, action: {:?}", top_state, current_action);

            match current_action {
                Action::Shift(new_state) => {
                    let token = lookahead.unwrap();
                    let node = tree.create_node(GrammarToken::Terminal(lookahead_type.unwrap()), token.text.span, Some(token.text), None);
                    stack.push((new_state, node));
                    next_index = self.advance_token(tokens, next_index + 1);
                },
                Action::Reduce(rule_i) => {
                    let rule = self.rules[rule_i as usize];

                    let initial_span = stack.last()
                        .map(|(_state, node)| tree.node(*node).span)
                        .unwrap_or_else(SpanLoc::zero);

                    // Reduce the last (rule.1) nodes into a NonTerminal node of type (rule.0)
                    // So first we create the node
                    let reduced_node = tree.create_node(GrammarToken::NonTerminal(rule.0), initial_span, None, None);

                    // Then we add all the rule.1 last nodes in the stack in reverse
                    // Why reverse? we're iterating the grammar Left to Right, so if we iterate the
                    // stack we will visit last, last - 1, last - 2... so Right to Left, that is a
                    // problem when we're constructing nodes (because they will be reversed)
                    // that is also why we separate the node parent assignment and the stack popping
                    for i in (0..rule.1 as usize).rev() {
                        let (_state, node) = stack[stack.len() - i - 1];
                        tree.reassign_parent(node, Some(reduced_node));
                    }

                    // Pop rule.1 states/nodes (we already consumed them into the nonterminal node)
                    for _ in 0..rule.1 {
                        stack.pop();
                    }

                    let curr_top = stack.last().unwrap().0;
                    // Find the new state trough the goto table
                    let new_state = self.get_goto(curr_top, rule.0);
                    // Push the new state with the new reduced node (that has the previous nodes as
                    // childrens).
                    stack.push((new_state, reduced_node));
                },
                Action::Accept => return Ok(tree),
                Action::Reject => {
                    //panic!("Input rejected on index {} (tk: {:?})", next_index, lookahead)
                    return Err(ParsingError {
                        token: lookahead.map(|x| x.ttype),
                        token_loc: lookahead.map(|x| x.text.span)
                            .unwrap_or_else(|| {
                                tokens.last()
                                    .map_or_else(SpanLoc::zero, |x| x.text.span.just_after())
                            }),
                        expected: self.find_possible_tokens(top_state),
                    })
                },
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LRConflict<T: ScannerTokenType, N: GrammarTokenType> {
    Goto(u32, N, u32, u32),
    Action(u32, Option<T>, Action, Action),
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash, Ord, PartialOrd)]
enum ParsingRule {
    Root, Grammar(GrammarRuleIndex),
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash, Ord, PartialOrd)]
struct ParsingItem {
    rule: ParsingRule,
    pos: u32,
}

// TODO: use hashableset?
#[derive(Clone, Eq, PartialEq, Debug, Hash)]
struct ParsingItemSet(Vec<ParsingItem>);

#[derive(Clone)]
struct DfaBuildStateData<T: ScannerTokenType, N: GrammarTokenType> {
    item_set: ParsingItemSet,
    edges: Vec<(u32, GrammarToken<T, N>)>,
}

struct DfaBuildData<'a, T: ScannerTokenType, N: GrammarTokenType> {
    grammar: &'a Grammar<T, N>,
    state_to_index: HashMap<ParsingItemSet, u32>,
    states: Vec<DfaBuildStateData<T, N>>,// TODO: think of some way to use a reference.
}

impl<'a, T, N> DfaBuildData<'a, T, N> where T: ScannerTokenType + Enumerable + Hash + 'static, N: GrammarTokenType  + 'static {
    fn get_expected_token(&self, item: &ParsingItem) -> Option<GrammarToken<T, N>> {
        match item.rule {
            ParsingRule::Root => {
                if item.pos == 0 {
                    Some(self.grammar.root)
                } else {
                    None
                }
            },
            ParsingRule::Grammar(rule) => {
                self.grammar.get_rule(rule).get(item.pos as usize).cloned()
            },
        }
    }

    fn closure(&self, items: &mut ParsingItemSet) {
        // original def. of closure(I)
        // ---
        // Every item in I is also an item in Closure(I)
        // If A→α•B β is in Closure(I) and B→γ is an item, then add B→•γ to Closure(I)
        // Repeat until no more new items can be added to Closure(I)
        // ---
        // Now I mutate the original set so the first step is already done for us,
        // We just need to check what items can be added
        // Quick note: every item generates the same items so there's no point in retrying with the
        // old items, that's why we iterate the items only once.

        let mut curr_i = 0usize;
        while let Some(curr) = items.0.get(curr_i) {
            // Prepare index for next loop (we won't be using it inside the loop)
            curr_i += 1;

            let next_rule_item = match self.get_expected_token(curr) {
                Some(GrammarToken::NonTerminal(token)) => {
                    token
                },
                _ => continue,
            };

            let definitions = &self.grammar.defs[next_rule_item.index()];
            for (index, _def) in definitions.iter().enumerate() {
                let new_item = ParsingItem {
                    rule: ParsingRule::Grammar(GrammarRuleIndex(next_rule_item.index() as u16, index as u16)),
                    pos: 0
                };
                if !items.0.contains(&new_item) {
                    items.0.push(new_item);
                }
            }
        }
        items.0.sort();
    }

    fn goto(&self, from: &ParsingItemSet, token: GrammarToken<T, N>) -> ParsingItemSet {
        // What appens to itemset "from" after it consumes token?
        // To find it just consume the token in every rule (or skip the rule if it doesn't present
        // the same token) then closure the found itemset.
        let mut to_set = ParsingItemSet(Vec::new());

        for x in from.0.iter() {
            match self.get_expected_token(x) {
                Some(expected) if expected == token => {},
                _ => continue,
            }

            let next_item = ParsingItem {
                rule: x.rule,
                pos: x.pos + 1
            };

            if !to_set.0.contains(&next_item) {
                to_set.0.push(next_item);
            }
        }
        self.closure(&mut to_set);

        to_set
    }

    fn create_state(&mut self, set: ParsingItemSet) -> u32 {
        let new_id = self.states.len() as u32;
        self.state_to_index.insert(set.clone(), new_id);
        self.states.push(DfaBuildStateData {
            item_set: set,
            edges: vec![]
        });
        new_id
    }

    fn build_dfa(&mut self) {
        // page 11+ of https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-035-computer-language-engineering-spring-2010/lecture-notes/MIT6_035S10_lec03b.pdf
        // The key idea is taken from there but the S state is a root state whose rule is:
        // S ::= <grammar.root> $
        // This simplifies the operation quite a bit (and shrinks the table by one state).

        self.states.clear();
        self.state_to_index.clear();

        let mut root_set = ParsingItemSet(vec![
            ParsingItem {
                rule: ParsingRule::Root,
                pos: 0,
            }
        ]);
        self.closure(&mut root_set);

        self.create_state(root_set);

        let mut next_index = 0usize;

        while let Some(state) = self.states.get(next_index).cloned() {// TODO: remove clone
            // Prepare index for next loop, it won't be used in this.
            next_index += 1;
            for item in state.item_set.0.iter() {
                let expected_token = self.get_expected_token(item);
                if let Some(x) = expected_token {
                    let new_state = self.goto(&state.item_set, x);
                    let goto_index = match self.state_to_index.get(&new_state) {
                        Some(x) => {
                            //eprintln!("{} + {:?} = {}", next_index - 1, expected_token, *x);
                            *x
                        },
                        None => {
                            self.create_state(new_state.clone())
                        }
                    };
                    //println!("{} + {:?} = [{}]{:?}", next_index - 1, x, goto_index, new_state);
                    self.states[next_index - 1].edges.push((goto_index, x));
                }
            }
        }
    }

    fn first(&self, tokens: &[GrammarToken<T, N>]) -> HashSet<Option<T>> {
        let mut res = HashSet::new();
        if tokens.is_empty() {
            res.insert(None);
            return res
        }
        let mut can_be_0 = false;
        for token in tokens {
            can_be_0 = false;
            match token {
                GrammarToken::Terminal(x) => {
                    res.insert(Some(*x));
                },
                GrammarToken::NonTerminal(x) => {
                    let rules = &self.grammar.defs[x.index()];
                    for rule in rules {
                        // TODO: resolve recursion (specifically left recursion)
                        // How can I trigger this recursion?
                        // Found out you can by trying to compile this grammar:
                        // <A> = <B>
                        // <B> = <A> <B>
                        let mut rule_tks = self.first(rule);
                        can_be_0 |= rule_tks.remove(&None);
                        res.extend(rule_tks);
                    }
                },
            }
            if !can_be_0 {
                break
            }
        }

        if can_be_0 {
            res.insert(None);
        }

        res
    }

    fn build_follow(&self) -> Vec<HashSet<Option<T>>> {
        // This function builds the "function" Follow(NT) -> Set<Option<T>>
        // It's explained in further detail here: https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-035-computer-language-engineering-spring-2010/lecture-notes/MIT6_035S10_lec03b.pdf
        // The idea is this, Follow(X) tells what terminal tokens the grammar could accept after
        // X, to do this we traverse all of the grammar in search for rules like
        // A -> a X b
        // Then the grammar might accept the first token of whatever b might derive into or, if
        // b might also be an empty sequence, Follow(A)
        // Since this is recursive it's easier to compute the function for all NonTokens, updating
        // every NT set until nothing changes anymore.
        // (Note: even if you see three nested for loops this is quite efficient since it could be
        // written as loop { traverse the grammar searching for rule "A -> a X b" }, the nested loops
        // are because of how the grammar is written into memory (a Vec<Vec<Vec<GrammarToken>>>))

        let mut res = vec![HashSet::new(); N::enumerate().len()];
        let mut is_changed = true;

        match self.grammar.root {
            GrammarToken::Terminal(_) => {},
            GrammarToken::NonTerminal(token) => {
                // Take into account the rule
                // R -> <grammar.root> $
                res[token.index()].insert(None);
            },
        }

        while is_changed {
            is_changed = false;

            for (def_i, def) in self.grammar.defs.iter().enumerate() {
                for rule in def.iter() {
                    for (token_i, token) in rule.iter().enumerate() {
                        let token = match token {
                            GrammarToken::Terminal(_) => continue,
                            GrammarToken::NonTerminal(x) => x,
                        };
                        // Rule is A -> a B b
                        // where A and B are nonterminals and a and b are token sequences (NT or T).

                        let after_tokens = &rule[token_i + 1..];
                        let mut firsts = self.first(after_tokens);
                        let firsts_can_be_0 = firsts.remove(&None);

                        // This is Follow(B)
                        let target = &mut res[token.index()];
                        let pre_len = target.len();

                        target.extend(firsts);
                        if firsts_can_be_0 {
                            // Then b is empty (or it can derive an empty sequence)
                            let i2 = index_twice(res.as_mut_slice(), token.index(), def_i);
                            if let IndexTwice::Both(target, from) = i2 {
                                target.extend(from.iter())
                            }
                        }
                        is_changed |= pre_len != res[token.index()].len()
                    }
                }
            }
        }
        res
    }

    fn build_table(&self) -> Result<ShiftReducer<T, N>, LRConflict<T, N>> {
        // The Follow(t: NonTerminal) -> Set<Option<Terminal>> function
        // It's easier to build it in advance than to deal with recursion...
        let follow = self.build_follow();

        let rules_index: Vec<_> = self.grammar.defs.iter().enumerate().flat_map(|(def_i, def)| {
            let defi = def_i as u16;
            (0..def.len()).map(move |rule_i| { GrammarRuleIndex(defi, rule_i as u16) })
        }).collect();

        let rules: Vec<_> = self.grammar.defs.iter().enumerate().flat_map(|(def_i, def)| {
            let token = N::enumerate().nth(def_i).unwrap();
            def.iter().map(move |rule| {
                (token, rule.len() as u32)
            })
        }).collect();

        let action_len = self.states.len() * (T::enumerate().len() + 1);
        let goto_len = self.states.len() * N::enumerate().len();

        let mut res = ShiftReducer {
            root_node_type: self.grammar.root,
            state_count: self.states.len() as u32,
            action_table: vec![Action::Reject; action_len],
            goto_table: vec![u32::max_value(); goto_len],
            rules,
            ignore_tokens: self.grammar.ignored.clone(),
        };

        for (state_i, state) in self.states.iter().enumerate() {
            for item in state.item_set.0.iter() {
                let next_token = self.get_expected_token(item);
                if next_token.is_none() {
                    match item.rule {
                        ParsingRule::Root => {
                            // Accept Rule "Root -> <grammar.root> * $"
                            // If we are at point * and there is no more input
                            res.set_action(state_i as u32, None, Action::Accept)?;
                        },
                        ParsingRule::Grammar(rule_i) => {
                            // The rule is R -> a ... b *
                            // where * is the position, so we can reduce everything to R
                            // But should we? *Veritasium noises*
                            // To not create conflicts we use the Follow function to check
                            // with what tokens the grammar can advance after the reduction
                            // so we use the provided token as a lookahead
                            // (here's where the name LR(1) comes from).

                            let reduce_tokens = &follow[rule_i.0 as usize];

                            // Search the grammar rule into our, serialized rules
                            let rule_i = rules_index.iter()// TODO: use binary search
                                .position(|x| *x == rule_i)
                                .expect("Cannot find used rule in grammar") as u32;

                            // Write the reduction only on the lookahead tokens.
                            for x in reduce_tokens {
                                res.set_action(state_i as u32, *x, Action::Reduce(rule_i))?;
                            }
                        },
                    }
                }
            }
            for (to, token) in state.edges.iter() {
                match token {
                    GrammarToken::Terminal(token) => {
                        res.set_action(state_i as u32, Some(*token), Action::Shift(*to))?
                    },
                    GrammarToken::NonTerminal(token) => {
                        res.set_goto(state_i as u32, *token, *to)?
                    },
                }
            }
        }
        Ok(res)
    }
}

impl<T: ScannerTokenType + Enumerable + Hash + 'static, N: GrammarTokenType + 'static> Grammar<T, N> {
    pub fn to_ll_table(&self) -> Result<ShiftReducer<T, N>, LRConflict<T, N>> {
        let mut data = DfaBuildData {
            grammar: self,
            state_to_index: HashMap::new(),
            states: Vec::new()
        };
        data.build_dfa();
        data.build_table()
    }
}


#[cfg(test)]
mod tests {
    use std::iter::Cloned;
    use std::slice::Iter;

    use crate::parser::{Grammar, GrammarToken, SyntaxNode, SyntaxTree};
    use crate::scanner::{Token, TokenLoc};
    use coldpiler_util::Enumerable;
    use crate::loc::SpanLoc;

    #[test]
    fn basic_test() {
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        enum TestTokenType {
            OPEN,
            CLOSE,
        }

        impl Enumerable for TestTokenType {
            type Iterator = Cloned<Iter<'static, Self>>;

            fn index(&self) -> usize {
                match self  {
                    TestTokenType::OPEN => 0,
                    TestTokenType::CLOSE => 1,
                }
            }

            fn enumerate() -> Self::Iterator {
                use TestTokenType::*;
                static TYPES: [TestTokenType; 2] = [OPEN, CLOSE];
                TYPES.iter().cloned()
            }
        }

        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        enum TestGrammarTokenType {
            X,
        }

        type TGTT = TestGrammarTokenType;

        impl Enumerable for TestGrammarTokenType {
            type Iterator = Cloned<Iter<'static, Self>>;

            fn index(&self) -> usize {
                match self {
                    TGTT::X => 0,
                }
            }

            fn enumerate() -> Self::Iterator {
                static TYPES: [TGTT; 1] = [TGTT::X];
                TYPES.iter().cloned()
            }
        }

        use GrammarToken::Terminal as T;
        use GrammarToken::NonTerminal as NT;
        use TestTokenType::*;
        use TestGrammarTokenType::*;
        let grammar = Grammar::from_raw(NT(X), vec![
            vec![vec![T(OPEN), NT(X), T(CLOSE)],
                 vec![T(OPEN), T(CLOSE)]]
        ], vec![]);
        // Grammar is:
        // X ::= ( X ) | ( )

        let table = grammar.to_ll_table().unwrap();
        let tokens = vec![
            Token { text: TokenLoc::of(0, 0, 0, 0, 0), ttype: OPEN },
            Token { text: TokenLoc::of(0, 0, 1, 0, 1), ttype: OPEN },
            Token { text: TokenLoc::of(1, 0, 2, 0, 2), ttype: CLOSE },
            Token { text: TokenLoc::of(1, 0, 3, 0, 3), ttype: CLOSE },
        ];
        let tree = table.parse(&tokens).unwrap();
        assert_eq!(SyntaxTree::from_nodes_unchecked(vec![
            SyntaxNode { parent: Some(5), gtype: T(OPEN), text: Some(TokenLoc::of(0, 0, 0, 0, 0)), span: SpanLoc::of(0, 0, 0, 0), children: vec![] },
            SyntaxNode { parent: Some(3), gtype: T(OPEN), text: Some(TokenLoc::of(0, 0, 1, 0, 1)), span: SpanLoc::of(0, 1, 0, 1), children: vec![] },
            SyntaxNode { parent: Some(3), gtype: T(CLOSE), text: Some(TokenLoc::of(1, 0, 2, 0, 2)), span: SpanLoc::of(0, 2, 0, 2), children: vec![] },
            SyntaxNode { parent: Some(5), gtype: NT(X), text: None, span: SpanLoc::of(0, 1, 0, 2), children: vec![1, 2] },
            SyntaxNode { parent: Some(5), gtype: T(CLOSE), text: Some(TokenLoc::of(1, 0, 3, 0, 3)), span: SpanLoc::of(0, 3, 0, 3), children: vec![]},
            SyntaxNode { parent: None, gtype: NT(X), text: None, span: SpanLoc::of(0, 0, 0, 3), children: vec![0, 3, 4] },
        ]), tree);
    }

    #[test]
    fn harder_test() {
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        enum TestTokenType {
            A,
            B,
        }

        impl Enumerable for TestTokenType {
            type Iterator = Cloned<Iter<'static, Self>>;

            fn index(&self) -> usize {
                use TestTokenType::*;
                match self  {
                    A => 0,
                    B => 1,
                }
            }

            fn enumerate() -> Self::Iterator {
                use TestTokenType::*;
                static TYPES: [TestTokenType; 2] = [A, B];
                TYPES.iter().cloned()
            }
        }

        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        enum TestGrammarTokenType {
            X,
        }

        type TGTT = TestGrammarTokenType;

        impl Enumerable for TestGrammarTokenType {
            type Iterator = Cloned<Iter<'static, Self>>;

            fn index(&self) -> usize {
                match self {
                    TGTT::X => 0,
                }
            }

            fn enumerate() -> Self::Iterator {
                static TYPES: [TGTT; 1] = [TGTT::X];
                TYPES.iter().cloned()
            }
        }

        use GrammarToken::Terminal as T;
        use GrammarToken::NonTerminal as NT;
        use TestTokenType::*;
        use TestGrammarTokenType::*;
        let grammar = Grammar::from_raw(NT(X), vec![
            vec![vec![T(A)],
                 vec![T(A), T(B)]]
        ], vec![]);
        // Grammar is:
        // X ::= A B | A
        // This requires a lookahead when reducing

        let table = grammar.to_ll_table().unwrap();
        let tokens = vec![
            Token { text: TokenLoc::of(0, 0, 0, 0, 0), ttype: A },
            Token { text: TokenLoc::of(1, 0, 1, 0, 1), ttype: B },
        ];
        let tree = table.parse(&tokens);
        assert_eq!(SyntaxTree::from_nodes_unchecked(vec![
            SyntaxNode { parent: Some(2), gtype: T(A), text: Some(TokenLoc::of(0, 0, 0, 0, 0)), span: SpanLoc::of(0, 0, 0, 0), children: vec![] },
            SyntaxNode { parent: Some(2), gtype: T(B), text: Some(TokenLoc::of(1, 0, 1, 0, 1)), span: SpanLoc::of(0, 1, 0, 1), children: vec![] },
            SyntaxNode { parent: None, gtype: NT(X), text: None, span: SpanLoc::of(0, 0, 0, 1), children: vec![0, 1] },
        ]), tree.unwrap());
    }


    #[test]
    fn calculator_test() {
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        enum TestTokenType {
            NUMBER, PLUS, SPACE
        }

        impl Enumerable for TestTokenType {
            type Iterator = Cloned<Iter<'static, Self>>;

            fn index(&self) -> usize {
                match self  {
                    TestTokenType::NUMBER => 0,
                    TestTokenType::PLUS => 1,
                    TestTokenType::SPACE => 2,
                }
            }

            fn enumerate() -> Self::Iterator {
                use TestTokenType::*;
                static TYPES: [TestTokenType; 3] = [NUMBER, PLUS, SPACE];
                TYPES.iter().cloned()
            }
        }

        #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
        enum TestGrammarTokenType {
            Statement,
        }

        type TGTT = TestGrammarTokenType;

        impl Enumerable for TestGrammarTokenType {
            type Iterator = Cloned<Iter<'static, Self>>;

            fn index(&self) -> usize {
                match self {
                    TGTT::Statement => 0,
                }
            }

            fn enumerate() -> Self::Iterator {
                static TYPES: [TGTT; 1] = [TGTT::Statement];
                TYPES.iter().cloned()
            }
        }

        use GrammarToken::Terminal as T;
        use GrammarToken::NonTerminal as NT;
        use TestTokenType::*;
        use TestGrammarTokenType::*;
        let grammar = Grammar::from_raw(
            NT(Statement),
            vec![
                vec![vec![T(NUMBER)], vec![T(NUMBER), T(PLUS), NT(Statement)]],
            ],
            vec![]
        );
        // Grammar is:
        // S ::= Number | Number Plus <S>

        let table = grammar.to_ll_table().unwrap();
        // Original text: 2+40
        let tokens = vec![
            Token { text: TokenLoc::of(0, 0, 0, 0, 0), ttype: NUMBER },
            Token { text: TokenLoc::of(1, 0, 1, 0, 1), ttype: PLUS },
            Token { text: TokenLoc::of(2, 0, 2, 0, 3), ttype: NUMBER },
        ];
        let tree = table.parse(&tokens);
        eprintln!("{:?}", tree);
    }

    #[test]
    fn fking_long_test() {
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        enum TestTokenType {
            Eq, Var, If, Else, Fun, BoolLiteral, NumberLiteral, Identifier, ExprSeparator, OpenBrack, CloseBrack, OpenPhar, ClosePhar, Colon, Comma, Space
        }

        impl Enumerable for TestTokenType {
            type Iterator = Cloned<Iter<'static, Self>>;

            fn index(&self) -> usize {
                use TestTokenType::*;
                match self {
                    Eq => 0,
                    Var => 1,
                    If => 2,
                    Else => 3,
                    Fun => 4,
                    BoolLiteral => 5,
                    NumberLiteral => 6,
                    Identifier => 7,
                    ExprSeparator => 8,
                    OpenBrack => 9,
                    CloseBrack => 10,
                    OpenPhar => 11,
                    ClosePhar => 12,
                    Colon => 13,
                    Comma => 14,
                    Space => 15,
                }
            }

            fn enumerate() -> Self::Iterator {
                use TestTokenType::*;
                static TYPES: [TestTokenType; 16] = [Eq, Var, If, Else, Fun, BoolLiteral, NumberLiteral, Identifier, ExprSeparator, OpenBrack, CloseBrack, OpenPhar, ClosePhar, Colon, Comma, Space];
                TYPES.iter().cloned()
            }
        }

        #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
        enum TestGrammarTokenType {
            Program, FunctionDeclaration, FunctionArgsDeclaration, FunctionArgsDeclarationEntry, FunctionReturn, Block,
            BlockEntry, ExprOrDecl, Declaration, Expr, ExprOp, ExprBase, Assign, IfExpr, IfTail, FunctionCall,
            FunctionCallArgs, FunctionCallArgsEntry, Lit
        }

        type TGTT = TestGrammarTokenType;

        impl Enumerable for TestGrammarTokenType {
            type Iterator = Cloned<Iter<'static, Self>>;

            fn index(&self) -> usize {
                use TestGrammarTokenType::*;
                match self {
                    Program => 0,
                    FunctionDeclaration => 1,
                    FunctionArgsDeclaration => 2,
                    FunctionArgsDeclarationEntry => 3,
                    FunctionReturn => 4,
                    Block => 5,
                    BlockEntry => 6,
                    ExprOrDecl => 7,
                    Declaration => 8,
                    Expr => 9,
                    ExprOp => 10,
                    ExprBase => 11,
                    Assign => 12,
                    IfExpr => 13,
                    IfTail => 14,
                    FunctionCall => 15,
                    FunctionCallArgs => 16,
                    FunctionCallArgsEntry => 17,
                    Lit => 18,
                }
            }

            fn enumerate() -> Self::Iterator {
                use TestGrammarTokenType::*;

                static TYPES: [TGTT; 19] = [Program, FunctionDeclaration, FunctionArgsDeclaration, FunctionArgsDeclarationEntry, FunctionReturn, Block,
                    BlockEntry, ExprOrDecl, Declaration, Expr, ExprOp, ExprBase, Assign, IfExpr, IfTail, FunctionCall,
                    FunctionCallArgs, FunctionCallArgsEntry, Lit];
                TYPES.iter().cloned()
            }
        }

        use GrammarToken::Terminal as T;
        use GrammarToken::NonTerminal as NT;
        use TestTokenType::*;
        use TestGrammarTokenType::*;
        let grammar = Grammar::from_raw(
            NT(Program),
            vec![
                vec![
                    vec![NT(FunctionDeclaration), NT(Program)]
                ],
                vec![
                    vec![T(Fun), T(Identifier), NT(FunctionArgsDeclaration), NT(FunctionReturn), NT(Block)]
                ],
                vec![
                    vec![T(OpenPhar), T(ClosePhar)],
                    vec![T(OpenPhar), NT(FunctionArgsDeclarationEntry), T(ClosePhar)]
                ],
                vec![
                    vec![T(Identifier), T(Colon), T(Identifier)],
                    vec![T(Identifier), T(Colon), T(Identifier), T(Comma), NT(FunctionArgsDeclarationEntry)]
                ],
                vec![
                    vec![T(Colon), T(Identifier)],
                    vec![]
                ],
                vec![
                    vec![T(OpenBrack), NT(BlockEntry), T(CloseBrack)]
                ],
                vec![
                    vec![NT(ExprOrDecl)],
                    vec![NT(ExprOrDecl), T(ExprSeparator), NT(BlockEntry)]
                ],
                vec![
                    vec![NT(Expr)],
                    vec![NT(Declaration)],
                    vec![]
                ],
                vec![
                    vec![T(Var), T(Identifier), T(Eq), NT(Expr)]
                ],
                vec![
                    vec![NT(ExprOp)],
                    vec![T(Identifier), T(Eq), NT(Expr)]
                ],
                vec![
                    vec![NT(ExprBase)],
                    vec![NT(ExprOp), T(Identifier), NT(ExprBase)]
                ],
                vec![
                    vec![T(Identifier)], vec![NT(Block)], vec![NT(Lit)], vec![NT(IfExpr)], vec![NT(FunctionCall)]
                ],
                vec![
                    vec![T(Identifier), T(Eq), NT(Expr)]
                ],
                vec![
                    vec![T(If), NT(Expr), NT(Block), NT(IfTail)]
                ],
                vec![
                    vec![T(Else), NT(Block)], vec![]
                ],
                vec![
                    vec![T(Identifier), T(OpenPhar), NT(FunctionCallArgs), T(ClosePhar)]
                ],
                vec![
                    vec![T(OpenPhar), T(ClosePhar)],
                    vec![T(OpenPhar), NT(FunctionCallArgsEntry), T(ClosePhar)]
                ],
                vec![
                    vec![NT(Expr)], vec!
                    [NT(Expr), NT(FunctionCallArgsEntry)]
                ],
                vec![
                    vec![T(NumberLiteral)],
                    vec![T(BoolLiteral)]
                ]
            ],
            vec![Space]
        );
        // Grammar is:
        // S ::= Number | Number Plus <S>

        let table = grammar.to_ll_table().unwrap();
        //eprintln!("{:?}", table);
    }
}
