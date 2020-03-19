# Coldpiler

This is a more serious version of [compiler-exercises](https://github.com/SnowyCoder/compiler-exercises), so it's a toy compiler written in rust.
I've yet to decide what to compile so for now it's a really complicated calculator with compile time LALR table and regex compilation
(and optimization because I hate myself).
<br>
To make this project a bit more readable I've divided the project in three parts:
- `coldpiler_parser` contains the boilerplate code for the lexer and parser
- `coldpiler_codegen` provides the procedural macros that generate the code for lexer and parser
- `coldpiler` is the heart of the calculator (~70 lines)

### Dependencies
My plan is to go to 0-dependencies but if you want to use procedural macros `syn` and `quote` are almost required,
so I consider them (and `proc_macro2`) as part of the standard library.
This is not because of lazyness but because they just provide a good way to remove boilerplate code and they fall out of the scope of this project.

### Comments
I've tried to comment the strangest parts of the code but I should come back and explain more of what's going on.
That said the strongly-typed nature of rust helps understand more than python, so I'd use this project over the python one
to comprehend what I've done.

### Future
I plan to transform this project into a real compiler for my own language.
The plan is not to use any dependency and to write any nontrivial algorithm/data structure myself.
I think I'll provide compilation to RISC-V or WebAssembly (still to decide), and maybe interpretation.
