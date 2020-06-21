use coldpiler_codegen::coldpile;
use coldpiler_parser::scanner::{Token, TokenLoc};

use lang::*;

use crate::ast::{FunctionDefinition, FunctionSignature, Value};
use crate::ast::parse::SyntaxTree;
use crate::context::{Context, TextProvider};

mod context;
mod error;
mod ast;
mod register_allocator;
mod tac;
mod riscv;
mod interpret;

coldpile!(
lang = {
    <Program> = <FunctionDeclaration> <Program> | ;
    <FunctionDeclaration> = Fun Identifier <FunctionArgsDeclaration> <FunctionReturn> <Block>;
    <FunctionArgsDeclaration> = OpenPhar ClosePhar | OpenPhar <FunctionArgsDeclarationEntry> ClosePhar;
    <FunctionArgsDeclarationEntry> = Identifier Colon Identifier | Identifier Colon Identifier Comma <FunctionArgsDeclarationEntry>;
    <FunctionReturn> = Colon Identifier | ;
    <Block> = OpenBrack <BlockEntry> CloseBrack;
    <BlockEntry> = <ExprOrDecl> | <ExprOrDecl> ExprSeparator <BlockEntry>;
    <ExprOrDecl> = <Expr> | <Declaration> | ;
    <Declaration> = Var Identifier Eq <Expr>;
    // Differentiate Expr, ExprOp and ExprBase to disambiguate
    <Expr> = <ExprOp> | Identifier Eq <Expr>;
    <ExprOp> = <ExprBase> | <ExprOp> Identifier <ExprBase>;
    <ExprBase> = Identifier | <Block> | <Lit> | <IfExpr> | <FunctionCall>;
    <Assign> = Identifier Eq <Expr>;
    <IfExpr> = If <Expr> <Block> <IfTail>;
    <IfTail> = Else <Block> | ;
    <FunctionCall> = Identifier <FunctionCallArgs>;
    <FunctionCallArgs> = OpenPhar <FunctionCallArgsEntry> ClosePhar | OpenPhar ClosePhar;
    <FunctionCallArgsEntry> = <Expr> | <Expr> Comma <FunctionCallArgsEntry>;
    <Lit> = NumberLiteral | BoolLiteral;
    Eq = "="
    Var = "var"
    If = "if"
    Else = "else"
    Fun = "fun"
    BoolLiteral = "true|false"
    NumberLiteral = "0b[01]+|0x[0-9a-fA-F]+|0o[0-7]+|[0-9]+"
    Identifier = r#"[a-zA-Z][0-9a-zA-Z_]*|[-+*~/!|^&<>]"#
    ExprSeparator = ";"
    OpenBrack = "[{]"
    CloseBrack = "}"
    OpenPhar = "[(]"
    ClosePhar = ")"
    Colon = ":"
    Comma = ","
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

pub fn run_analyze(context: &mut Context) -> Result<(), ()> {
    let errors = ast::analyzer::analyze_all(&mut context.ast);
    if !errors.is_empty() {
        for error in errors {
            context.print_error(&error);
        }
        Err(())
    } else {
        Ok(())
    }
}

pub fn print(tree: &SyntaxTree, nodei: usize) {
    let node = tree.node(nodei);
    eprintln!("{}: {:?}", nodei, node);
    for x in node.children.iter().copied() {
        print(tree, x);
    }
}

pub fn run(content: String) -> Result<Value, ()> {
    let mut context = Context::new(TextProvider::Plain(content));
    let tokens = run_tokenize(&mut context)?;
    //println!("{:?}", tokens);
    let st = run_parse(&mut context, tokens)?;

    ast::parse::build_file(&mut context, &st);

    run_analyze(&mut context)?;

    let main_id = context.ast.find_function(&FunctionSignature::of(context.bank.main, Vec::new())).expect("Cannot find main");
    let mut tac_prog = tac::ast_program_to_tac(&mut context.ast, main_id);

    for x in tac_prog.funs.iter() {
        eprintln!("{}", x);
    }
    /*for f in tac_prog.funs.iter_mut() {
        f.convert_out_of_ssa();
    }*/

    let riscv = riscv::emit_program(&tac_prog, &context.ast);
    for (index, fun) in riscv.iter().enumerate() {
        let name = context.ast.functions.iter().find_map(|f| match f {
            FunctionDefinition::Builtin(_) => None,
            FunctionDefinition::Custom(x) => if x.tac_id == index { Some(x) } else { None },
        }).unwrap().name;
        let name = context.get_text(name.0.trie_index);
        eprintln!("{}: ", name);
        //eprintln!("{:?}", fun);
        for i in fun.iter().copied() {
            eprintln!("\t{}\t", i.encode().to_str());
        }
    }
    //let ret = interpret::exec_expr(&mut interpret::SymbolTable::new(&context), main.body);

    Ok(Value::Unit)
}
