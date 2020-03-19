extern crate proc_macro;

use proc_macro::TokenStream;

use coldpiler_parser;
use coldpiler_parser::parser::Enumerable;
use coldpiler_parser::scanner::{NfaToDfaError, regex_map_to_nfa, RegexReport};
use syn::{Ident, parse_macro_input};

use quote::{quote, quote_spanned};

use crate::parse::{Definition, Grammar, GrammarDefinition, Grammars, GrammarToken};
use crate::token_types::{ParserPlaceholderType, ScannerPlaceholderType};
use proc_macro2::Span;

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
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub enum #enum_name {
            #(#token_names), *
        }


        impl coldpiler_parser::parser::Enumerable for #enum_name {
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

    let dfa = match nfa.unwrap().to_dfa() {
        Ok(x) => x,
        Err(e) => {
            let desc = match e {
                NfaToDfaError::StateConflict(a, b) => {
                    let an = data[a.index()].0;
                    let bn = data[b.index()].0;
                    format!("Conflict between  '{}' and '{}'", an, bn)
                },
            };
            return quote_spanned!(
                grammar.name.span() =>
                compile_error!(#desc);
                unimplemented!()
            )
        },
    };

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

    ParserPlaceholderType::set_size(data.len() as u32);

    let terminal_names: Vec<&Ident> = grammar.defs.iter().filter_map(|x| {
        match x {
            Definition::Grammar(_) => None,
            Definition::Scanner(x) => Some(&x.name),
        }
    }).collect();

    let nonterminal_names: Vec<&Ident> = data.iter().map(|x| &x.name).collect();

    let root_token = match grammar.defs.first().expect("Empty grammar") {
        Definition::Grammar(_) => coldpiler_parser::parser::GrammarToken::NonTerminal(ParserPlaceholderType(0)),
        Definition::Scanner(_) => coldpiler_parser::parser::GrammarToken::Terminal(ScannerPlaceholderType(0)),
    };

    let mut errors = Vec::new();

    let defmap: Vec<coldpiler_parser::parser::GrammarDefinition<ScannerPlaceholderType, ParserPlaceholderType>> = data.iter().map(|x| {
        x.rules.iter().map(|rule|
            rule.0.iter().map(|token|
                match token {
                    GrammarToken::Terminal(name) => {
                        match terminal_names.iter().position(|x| *x == name) {
                            Some(idx) => {
                                coldpiler_parser::parser::GrammarToken::Terminal(ScannerPlaceholderType(idx as u32))
                            },
                            None => {
                                errors.push((name.span(), format!("Cannot find Terminal {}", name)));
                                coldpiler_parser::parser::GrammarToken::Terminal(ScannerPlaceholderType(0))
                            }
                        }
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

    let create_parser_code = if errors.is_empty() {
        let real_grammar = coldpiler_parser::parser::Grammar::from_raw(
            root_token, defmap
        );
        let table = real_grammar.to_ll_table();
        let raw_table_code = table.to_raw_code(&terminal_names, &nonterminal_names);
        quote! {
            use ScannerTokenType as T;
            use ParserTokenType as N;
            ShiftReducer::from_raw(#raw_table_code)
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

        pub fn tokenize(src: &str) -> std::vec::Vec<coldpiler_parser::scanner::Token<ScannerTokenType>> {
            let dfa = ScannerTokenType::build_dfa();
            dfa.tokenize(src, 0)
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



#[cfg(test)]
mod tests {
    use coldpiler_parser::parser::{Grammar, GrammarToken};

    use crate::token_types::{ParserPlaceholderType, ScannerPlaceholderType};

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn it_not_works() {
        ScannerPlaceholderType::set_size(2);
        ParserPlaceholderType::set_size(2);
        let grammar = Grammar::from_raw(
            GrammarToken::NonTerminal(ParserPlaceholderType(0)),
            vec![
                vec![
                    vec![GrammarToken::Terminal(ScannerPlaceholderType(1)), GrammarToken::NonTerminal(ParserPlaceholderType(1))]
                ],
                vec![vec![GrammarToken::Terminal(ScannerPlaceholderType(0)), GrammarToken::Terminal(ScannerPlaceholderType(1)), GrammarToken::NonTerminal(ParserPlaceholderType(1))], vec![]]
            ]
        );
        grammar.to_ll_table();
    }
}
