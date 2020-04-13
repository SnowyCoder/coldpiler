#[macro_use]
extern crate lazy_static;
extern crate proc_macro;

use proc_macro::TokenStream;

use coldpiler_parser;
use coldpiler_parser::parser::LRConflict;
use coldpiler_parser::scanner::{regex_map_to_nfa, RegexReport};
use proc_macro2::Span;
use syn::{Ident, parse_macro_input};

use quote::{quote, quote_spanned};

use crate::parse::{Definition, Grammar, GrammarDefinition, Grammars, GrammarToken};
use crate::token_types::{ParserPlaceholderType, ScannerPlaceholderType};


mod parse;
mod token_types;

fn generate_token_code(enum_name: Ident, token_names: &[&Ident]) -> proc_macro2::TokenStream {
    let enum_index_defs = token_names.iter()
        .enumerate()
        .map(|(index, name)| {
            quote! {
                #enum_name::#name => #index,
            }
    });
    let type_count = token_names.len();
    quote! {
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub enum #enum_name {
            #(#token_names), *
        }


        impl coldpiler_util::Enumerable for #enum_name {
            type Iterator = core::iter::Cloned<core::slice::Iter<'static, Self>>;

            fn index(&self) -> usize {
                return match self {
                    #(#enum_index_defs)*
                }
            }

            fn enumerate() -> Self::Iterator {
                static TYPES: [#enum_name; #type_count] = [#(#enum_name::#token_names), *];
                TYPES.iter().cloned()
            }
        }
    }
}

fn generate_tokens_code(grammar: &Grammar) -> proc_macro2::TokenStream {
    let mut scanner_names = Vec::new();
    let mut parser_names = Vec::new();

    for def in grammar.defs.iter() {
        match def {
            Definition::Scanner(s) => {
                scanner_names.push(&s.name);
            },
            Definition::Grammar(g) => {
                parser_names.push(&g.name);
            },
        }
    }

    let scanner_token = generate_token_code(
        Ident::new("ScannerTokenType", Span::call_site()),
        &scanner_names
    );
    let parser_token = generate_token_code(
        Ident::new("ParserTokenType", Span::call_site()),
        &parser_names
    );

    quote! {
        #scanner_token
        #parser_token
    }
}

fn generate_scanner_code(grammar: &Grammar) -> proc_macro2::TokenStream {
    let data: Vec<(&Ident, String)> = grammar.defs.iter().filter_map(|x| {
        match x {
            Definition::Grammar(_) => None,
            Definition::Scanner(def) => {
                Some((&def.name, def.rule.value()))
            },
        }
    }).collect();

    let names: Vec<String> = data.iter().map(|x| format!("{}", x.0)).collect();
    ScannerPlaceholderType::set_debug_names(names);

    let work_data: Vec<(ScannerPlaceholderType, &str)> = data.iter()
        .enumerate()
        .map(|(index, x)| {
            (ScannerPlaceholderType(index as u32), x.1.as_ref())
        })
        .collect();
    ScannerPlaceholderType::set_size(work_data.len() as u32);

    let mut report = RegexReport::new();
    let nfa = regex_map_to_nfa(&mut report, &work_data);

    if !report.is_empty() {
        let errors= report.iter().map(|x| {
            let span = grammar.defs.iter().filter_map(|def| match def {
                Definition::Grammar(_) => None,
                Definition::Scanner(t) => Some(t),
            }).nth(x.regex_entry).unwrap().rule.span();
            let msg = &x.description;
            quote_spanned!(
                span =>
                compile_error!(#msg);
            )
        });
        return quote! {
            #(#errors)*
            unimplemented!()
        }
    }

    let dfa = nfa.unwrap().to_dfa();
    let (dfa_raw1, dfa_raw2) = dfa.minimize_hopcroft().into_raw();

    let dfa_raw_1_code = dfa_raw1.iter().map(|x| {
       match x {
           None => quote!(None),
           Some(index) => {
               let type_name = data[index.0 as usize].0;
               quote! { Some(ScannerTokenType::#type_name) }
           },
       }
    });

    let dfa_raw_2_code = dfa_raw2.iter().map(|x| {
        match x {
            None => quote!(None),
            Some(index) => quote!(Some(#index)),
        }
    });

    let dfa_raw_code = quote! {
        vec![#(#dfa_raw_1_code), *],
        vec![#(#dfa_raw_2_code), *]
    };

    quote! {
        coldpiler_parser::scanner::Scanner::from_raw(#dfa_raw_code)
    }
}

fn generate_parser_code(grammar: &Grammar) -> proc_macro2::TokenStream {
    let data: Vec<&GrammarDefinition> = grammar.defs.iter().filter_map(|x| {
        match x {
            Definition::Grammar(x) => Some(x),
            Definition::Scanner(_) => None,
        }
    }).collect();

    let names: Vec<String> = data.iter().map(|x| format!("{}", x.name)).collect();
    ParserPlaceholderType::set_debug_names(names);

    ParserPlaceholderType::set_size(data.len() as u32);

    let terminals = grammar.defs.iter().filter_map(|x| {
        match x {
            Definition::Grammar(_) => None,
            Definition::Scanner(x) => Some(x),
        }
    });

    let terminal_names: Vec<&Ident> = terminals.clone().map(|x| &x.name).collect();

    let nonterminal_names: Vec<&Ident> = data.iter().map(|x| &x.name).collect();

    let root_token = match grammar.defs.first().expect("Empty grammar") {
        Definition::Grammar(_) => coldpiler_parser::parser::GrammarToken::NonTerminal(ParserPlaceholderType(0)),
        Definition::Scanner(_) => coldpiler_parser::parser::GrammarToken::Terminal(ScannerPlaceholderType(0)),
    };

    let mut errors = Vec::new();

    let find_terminal_by_name = |name: &Ident, errors: &mut Vec<(Span, String)>| {
        match terminal_names.iter().position(|x| *x == name) {
            Some(idx) => {
                ScannerPlaceholderType(idx as u32)
            },
            None => {
                errors.push((name.span(), format!("Cannot find Terminal {}", name)));
                ScannerPlaceholderType(0)
            }
        }
    };


    let defmap: Vec<coldpiler_parser::parser::GrammarDefinition<ScannerPlaceholderType, ParserPlaceholderType>> = data.iter().map(|x| {
        x.rules.iter().map(|rule|
            rule.0.iter().map(|token|
                match token {
                    GrammarToken::Terminal(name) => {
                        coldpiler_parser::parser::GrammarToken::Terminal(find_terminal_by_name(name, &mut errors))
                    },
                    GrammarToken::NonTerminal(name) => {
                        match nonterminal_names.iter().position(|x| *x == name) {
                            Some(idx) => {
                                coldpiler_parser::parser::GrammarToken::NonTerminal(ParserPlaceholderType(idx as u32))
                            },
                            None => {
                                errors.push((name.span(), format!("Cannot find NonTerminal {}", name)));
                                coldpiler_parser::parser::GrammarToken::NonTerminal(ParserPlaceholderType(0))
                            }
                        }
                    },
                }
            ).collect()
        ).collect()
    }).collect();

    let ignored: Vec<_> = terminals.filter_map(|x| {
        if x.meta.ignored {
            Some(find_terminal_by_name(&x.name, &mut errors))
        } else {
            None
        }
    }).collect();

    let create_parser_code = if errors.is_empty() {
        let real_grammar = coldpiler_parser::parser::Grammar::from_raw(
            root_token, defmap, ignored
        );
        let table = real_grammar.to_ll_table();
        match table {
            Ok(table) => {
                let raw_table_code = table.to_raw_code(&terminal_names, &nonterminal_names);
                quote! {
                    use ScannerTokenType as T;
                    use ParserTokenType as N;
                    ShiftReducer::from_raw(#raw_table_code)
                }
            },
            Err(conflict) => {
                let desc = match conflict {
                    LRConflict::Goto(state, token, old, new) => {
                        let old = &data[old as usize].name;
                        let new = &data[new as usize].name;
                        format!("Goto conflict in state: {} with token {:?}, old: {:?} new: {:?}", state, token, old, new)
                    },
                    LRConflict::Action(state, token, old, new) => {
                        format!("Action conflict in state: {} with token {:?}, old: {:?} new: {:?}", state, token, old, new)
                    },
                };
                quote! {
                    compile_error!(#desc);
                    unimplemented!()
                }
            },
        }
    } else {
        let errors = errors.drain(..).map(|(span, descr)| {
            quote_spanned! {
                span => compile_error!(#descr);
            }
        });
        quote! {
            #(#errors)*
            unimplemented!()
        }
    };

    quote! {
        use coldpiler_parser::parser::{ShiftReducer, Action, GrammarToken};

        pub fn create_shift_parser() -> ShiftReducer<ScannerTokenType, ParserTokenType> {
            #create_parser_code
        }

        pub fn tokenize(trie: &mut coldpiler_util::radix_tree::RadixTree<u8>, src: &str) -> (std::vec::Vec<coldpiler_parser::scanner::Token<ScannerTokenType>>, std::vec::Vec<coldpiler_parser::scanner::TokenLoc>) {
            let dfa = ScannerTokenType::build_dfa();
            dfa.tokenize(trie, src, 0)
            // TODO: use custom tokenizer code
        }
    }
}

fn generate_grammar_code(grammar: &Grammar) -> proc_macro2::TokenStream {
    let tokens = generate_tokens_code(grammar);

    let scanner = generate_scanner_code(grammar);
    let parser = generate_parser_code(grammar);

    let name = &grammar.name;

    quote! {
        mod #name {
            #tokens

            impl ScannerTokenType {
                pub fn build_dfa() -> coldpiler_parser::scanner::Scanner < Self > {
                    #scanner
                }
            }
            #parser
        }
    }
}


#[proc_macro]
pub fn coldpile(item: TokenStream) -> TokenStream {
    let ast: Grammars = parse_macro_input!(item as Grammars);

    let codes = ast.0.iter().map(generate_grammar_code);

    quote!(
        #(#codes)*
    ).into()
}
