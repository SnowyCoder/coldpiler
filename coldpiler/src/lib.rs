use coldpiler_codegen::coldpile;

mod context;
mod error;
mod ast;
mod interpret;

coldpile!(
lang = {
    <Program> = Fun Main <Block>;
    <Block> = OpenBrack <BlockEntry> CloseBrack;
    <BlockEntry> = <ExprOrDecl> | <ExprOrDecl> ExprSeparator <BlockEntry>;
    <ExprOrDecl> = <Expr> | <Declaration> | ;
    <Declaration> = Var Identifier Eq <Expr>;
    // Differentiate Expr, ExprOp and ExprBase to disambiguate
    <Expr> = <ExprOp> | Identifier Eq <Expr>;
    <ExprOp> = <ExprBase> | <ExprOp> Identifier <ExprBase>;
    <ExprBase> = Identifier | <Block> | <Lit> | <IfExpr> | <PrintExpr>;
    <Assign> = Identifier Eq <Expr>;
    <IfExpr> = If <Expr> <Block> <IfTail>;
    <IfTail> = Else <Block> | ;
    <PrintExpr> = Println OpenPhar <Expr> ClosePhar;
    <Lit> = NumberLiteral | BoolLiteral;
    Eq = "="
    Var = "var"
    If = "if"
    Else = "else"
    Fun = "fun"
    Main = "main"
    Println = "println"
    BoolLiteral = "true|false"
    NumberLiteral = "0b[01]+|0x[0-9a-fA-F]+|0o[0-7]+|[0-9]+"
    Identifier = r#"[a-zA-Z][0-9a-zA-Z_]*|[-+*~/!|^&<>]"#
    ExprSeparator = ";"
    OpenBrack = "[{]"
    CloseBrack = "}"
    OpenPhar = "[(]"
    ClosePhar = ")"
    Space = "[\\s]+" @ignore
}
);

use lang::*;
use crate::ast::Value;
use coldpiler_parser::scanner::{Token, TokenLoc};
use crate::context::{Context, TextProvider};
use crate::error::{CompilationError, ErrorLoc};
use crate::interpret::SymbolTable;

struct TokenizationError(TokenLoc);

impl CompilationError for TokenizationError {
    fn error_type(&self) -> String {
        "Token not recognized".to_owned()
    }

    fn loc(&self) -> ErrorLoc {
        ErrorLoc::SingleLocation(self.0.span)
    }

    fn summarize(&self) -> String {
        "Unexpected char".to_owned()
    }

    fn description(&self) -> String {
        "Cannot recognize any token".to_owned()
    }
}

pub fn run_tokenize(context: &mut Context) -> Result<Vec<Token<ScannerTokenType>>, ()> {
    let content = context.source.read_all();
    let (tokens, unrecognized) = tokenize(&mut context.trie, &content);

    if !unrecognized.is_empty() {
        for loc in unrecognized {
            context.print_error(&TokenizationError(loc));
        }
        Err(())
    } else {
        Ok(tokens)
    }
}

pub fn run(content: String) -> Result<Value, ()> {
    let mut context = Context::new(TextProvider::Plain(content));

    let parser = create_shift_parser();
    let tokens = run_tokenize(&mut context)?;
    //println!("{:?}", tokens);
    let st = parser.parse(&tokens);
    let ast = ast::parse::build_main(&context, &st);
    let ret = interpret::exec(SymbolTable::new(&context), &ast)?;

    Ok(ret)
}
