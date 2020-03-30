use coldpiler_codegen::coldpile;

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

pub mod st {
}

pub fn run(file: String) {
    let parser = create_shift_parser();
    let tokens = tokenize(&file);
    //println!("{:?}", tokens);
    let st = parser.parse(&tokens);
    let ast = ast::parse::build_main(&st);
    let ret = interpret::exec_block(Default::default(), &ast);

    match ret {
        Ok(x) => {
            print!("Returned: ");
            match x {
                Value::Unit => println!("Unit"),
                Value::I32(x) => println!("{}", x),
                Value::Bool(x) => println!("{}", x),
            }
        },
        Err(x) => {
            println!("{:?}", x)
        },
    }
}
