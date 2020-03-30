# Language
Here's everything related to the language design.
That means the principles to follow, the state of the language and the future of
 it.

## Principles
There are some major and some minor points that I'll follow.
<br>Main Objectives:
- **Fun**: I'm developing this compiler in my free time, if I don't learn nor I
enjoy myself then it's wasted time
- **Fast**: It should run on a logisim-simulated SPIR-V processor.

These minor points should still be taken into consideration, but they're just
a want-list that will be hopefully built in the future.
- **Static-Analysis**: I would like to play around a lot with the static
analyzer.
- **Compile/Run-Time blending**: This is an Idea that I've been playing around
for quite sometime, it is a pumped-up version of an optimizer.
- **Templates**: Well, it should explain itself, my objective is to write a
general println function without runtime polymorphism, maybe the CR-blend could
help.
- **Function Generalization**: This is quite a broad topic. The main idea is that
the programmer has the same power as the compiler, that means he can overwrite
operations (and their priorities), create loop-like functions and so.

## Current Language
```
<Program> = Fun Main <Block>;
<Block> = OpenBrack <BlockEntry> CloseBrack;
<BlockEntry> = <ExprOrDecl> | <ExprOrDecl> ExprSeparator <BlockEntry>;
<ExprOrDecl> = <Expr> | <Declaration> | ;
<Declaration> = Var Identifier Eq <Expr>;
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
```

Example program:
```
fun main {
  var a = 2;
  var b = 3 + 1 * 2;
  println(a + b * 3);
  if b > 5 {
    println(42);
    16
  }
}
```

Note that the 16 above is returned.
This uses rust's idea that the last expression result is returned (if no ; is
  inserted)

## Improvements
- Operation precedence (`2+3*2` = 8, not 10)
- Other functions besides main (and generalize the built-in ones)
- Static analysis
- Put the input tokens in a Trie (to save up memory and speed up comparisons)
- Real compilation
