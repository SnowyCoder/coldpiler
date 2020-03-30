use syn::{braced, Ident, LitStr, Result, Token, Error};
use syn::parse::{Parse, ParseStream};

pub enum GrammarToken {
    Terminal(Ident),
    NonTerminal(Ident),
}

impl Parse for GrammarToken {
    fn parse(input: ParseStream) -> Result<Self> {
        let next = input.lookahead1();
        let res = if next.peek(Token![<]) {
            input.parse::<Token![<]>()?;
            let name: Ident = input.parse()?;
            input.parse::<Token![>]>()?;
            GrammarToken::NonTerminal(name)
        } else {
            GrammarToken::Terminal(input.parse()?)
        };
        Ok(res)
    }
}

pub struct GrammarRule(pub Vec<GrammarToken>);

impl Parse for GrammarRule {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut tokens: Vec<GrammarToken> = vec![];

        loop {
            let next = input.lookahead1();
            if next.peek(Token![|]) || next.peek(Token![;]) {
                break
            }
            tokens.push(input.parse()?);
        }

        Ok(GrammarRule(tokens))
    }
}

pub struct GrammarDefinition {
    pub name: Ident,
    pub rules: Vec<GrammarRule>
}

impl Parse for GrammarDefinition {
    fn parse(input: ParseStream) -> Result<Self> {
        input.parse::<Token![<]>()?;
        let name: Ident = input.parse()?;
        input.parse::<Token![>]>()?;
        input.parse::<Token![=]>()?;

        let mut defs = vec![];
        loop {
            defs.push(input.parse::<GrammarRule>()?);

            let next = input.lookahead1();

            if next.peek(Token![;]) {
                input.parse::<Token![;]>()?;
                break
            }
            input.parse::<Token![|]>()?;
        }

        Ok(GrammarDefinition {
            name,
            rules: defs,
        })
    }
}

pub struct ScannerMeta {
    pub ignored: bool,
}

impl Parse for ScannerMeta {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut meta = ScannerMeta {
            ignored: false
        };
        let mut errors = Vec::new();

        while input.peek(Token![@]) {
            input.parse::<Token![@]>()?;
            let name: Ident = input.parse()?;

            match name.to_string().as_str() {
                "ignore" =>  {
                    meta.ignored = true;
                },
                _ => {
                    errors.push(Error::new_spanned(name.clone(), format!("Unknown annotation {}", name)))
                }
            }
        }
        if let Some(error) = errors.drain(..).fold(None, |acc: Option<Error>, x| {
            Some(match acc {
                Some(mut old_err) => {
                    old_err.combine(x);
                    old_err
                },
                None => x,
            })
        }) {
            return Err(error);
        }

        Ok(meta)
    }
}

pub struct ScannerDefinition {
    pub name: Ident,
    pub rule: LitStr,
    pub meta: ScannerMeta,
}

impl Parse for ScannerDefinition {
    fn parse(input: ParseStream) -> Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let rule: LitStr = input.parse()?;
        let meta: ScannerMeta = input.parse()?;
        Ok(ScannerDefinition { name, rule, meta })
    }
}

pub enum Definition {
    Grammar(GrammarDefinition),
    Scanner(ScannerDefinition),
}

impl Parse for Definition {
    fn parse(input: ParseStream) -> Result<Self> {
        let def = if input.peek(Token![<]) {
            Definition::Grammar(input.parse::<GrammarDefinition>()?)
        } else {
            Definition::Scanner(input.parse::<ScannerDefinition>()?)
        };
        Ok(def)
    }
}

impl Definition {
    fn parse_within(input: ParseStream) -> Result<Vec<Self>> {
        let mut defs = Vec::new();
        while !input.is_empty() {
            defs.push(input.parse::<Definition>()?);
        }
        Ok(defs)
    }
}

pub struct Grammar {
    pub name: Ident,
    pub defs: Vec<Definition>,
}

impl Parse for Grammar {
    fn parse(input: ParseStream) -> Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let content;
        braced!(content in input);

        let defs = content.call(Definition::parse_within)?;
        Ok(Grammar {
            name, defs
        })
    }
}

pub struct Grammars(pub Vec<Grammar>);

impl Parse for Grammars {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut defs = Vec::new();
        while !input.is_empty() {
            defs.push(input.parse::<Grammar>()?);
        }
        Ok(Grammars(defs))
    }
}

