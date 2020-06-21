use crate::tac::tacir::TacFunction;
use crate::ast::ast_data::AstData;
use crate::tac::ast_function_to_tac_ir;
use crate::ast::FunctionDefinition;

pub struct TacProgram {
    pub funs: Vec<TacFunction>,
    pub entry_point: usize,
}

pub fn ast_program_to_tac(ast: &mut AstData, entry_point: usize) -> TacProgram {
    let mut res = TacProgram {
        funs: vec![],
        entry_point: 0
    };
    for fid in 0..ast.functions.len() {
        let fdec = match ast.functions.get_mut(fid).unwrap() {
            FunctionDefinition::Builtin(_) => continue,
            FunctionDefinition::Custom(x) => x,
        };
        fdec.tac_id = res.funs.len();
        let tac_fun = ast_function_to_tac_ir(ast, fid);
        if fid == entry_point {
            res.entry_point = res.funs.len();
        }
        res.funs.push(tac_fun);
    }
    res
}


