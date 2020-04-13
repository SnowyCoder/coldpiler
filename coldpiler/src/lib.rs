use coldpiler_codegen::coldpile;
use coldpiler_parser::scanner::{Token, TokenLoc};

use lang::*;

use crate::ast::parse::SyntaxTree;
use crate::ast::Value;
use crate::context::{Context, TextProvider};
use crate::interpret::SymbolTable;

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

struct TokenizationError(TokenLoc);

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

pub fn run_parse(context: &mut Context, tokens: Vec<Token<ScannerTokenType>>) -> Result<SyntaxTree, ()> {
    match create_shift_parser().parse(&tokens) {
        Ok(x) => Ok(x),
        Err(x) => {
            context.print_error(&x);
            Err(())
        }
    }
}

pub fn run(content: String) -> Result<Value, ()> {
    let mut context = Context::new(TextProvider::Plain(content));
    let tokens = run_tokenize(&mut context)?;
    //println!("{:?}", tokens);
    let st = run_parse(&mut context, tokens)?;
    let ast = ast::parse::build_main(&context, &st);
    let ret = interpret::exec(SymbolTable::new(&context), &ast)?;

    Ok(ret)
}
