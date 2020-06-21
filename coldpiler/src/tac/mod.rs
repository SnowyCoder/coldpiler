

mod dominance;
mod live_range;
mod program;
mod ssa;
mod tacir;

pub use tacir::*;
pub use program::{TacProgram, ast_program_to_tac};

