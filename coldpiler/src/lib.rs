use coldpiler_codegen::coldpile;

coldpile!(
simple_language = {
    <Statement> = <Statement> Op Number | Number;
    Op = r"[+-]"
    Number = r"[0123456789]+"
    Space = r#"[\s\t]+"#
}
);


#[cfg(test)]
mod tests {
    use crate::*;
    use simple_language::*;
    use simple_language::ParserTokenType::*;
    use coldpiler_parser::parser::{SyntaxTree, SyntaxNode, GrammarToken};

    type T = ScannerTokenType;
    type NT = ParserTokenType;

    fn calc_tree(tree: &SyntaxTree<T, NT>, node: &SyntaxNode<T, NT>) -> i64 {
        match node.gtype {
            GrammarToken::NonTerminal(Statement) => {
                let number_index = if node.children.len() == 1 {
                    0
                } else {
                    2
                };
                let mut number = tree.node(node.children[number_index])
                    .text
                    .as_ref()
                    .unwrap()
                    .parse::<i64>()
                    .unwrap();

                if node.children.len() == 3 {
                    let op = &tree.node(node.children[1]).text;
                    let lhs = calc_tree(tree, tree.node(node.children[0]));
                    if op.as_ref().map_or(false, |x| x.as_str() == "+") {
                        number = lhs + number;
                    } else {
                        number = lhs - number;
                    }
                }
                number
            },
            _ => 0,
        }
    }

    #[test]
    fn it_works() {
        let parser = create_shift_parser();
        
        let tokens = tokenize("45-5+2");
        println!("{:?}", tokens);
        let ast = parser.parse(&tokens[..5]);
        println!("{:?}", ast);

        let res = calc_tree(&ast, ast.node(ast.find_root().unwrap()));

        assert_eq!(42, res);
        // The most complex calculator I've ever written so far
    }
}
