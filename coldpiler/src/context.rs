use coldpiler_util::radix_tree::RadixTree;
use std::path::PathBuf;
use std::fs::File;
use std::io::Read;
use crate::error::{ErrorLoc, CompilationError};

#[derive(Clone)]
pub struct Context {
    pub trie: RadixTree<u8>,
    pub source: TextProvider,
}

impl Context {
    pub fn new(source: TextProvider) -> Context {
        Context {
            trie: RadixTree::new(),
            source
        }
    }

    pub fn get_text(&self, index: u32) -> String {
        String::from_utf8(self.trie.find_key(index)).expect("Invalid source code")
    }

    fn print_error_loc(&self, loc: ErrorLoc) {
        let loc = match loc {
            ErrorLoc::NoLocation() => return,
            ErrorLoc::SingleLocation(loc) => loc,
            ErrorLoc::DoubleLocation(_, _) => unimplemented!(),
        };
        let same_line = loc.start_line == loc.end_line;
        let all_src = self.source.read_all();
        let lines = all_src.lines();

        if same_line {
            let (prev_line, mut lines) = if loc.start_line == 0 {
                (None, lines.skip(0))
            } else {
                let mut lines= lines.skip(loc.start_line as usize - 1);
                let line = lines.next().unwrap();
                (Some(line), lines)
            };

            let current = lines.next().unwrap();
            let next_line = lines.next();

            let line_loc_str = format!("{}", loc.start_line);

            if let Some(line) = prev_line {
                eprintln!(" {} |{}", " ".repeat(line_loc_str.len()), line);
            }
            eprintln!(" {} |{}", line_loc_str, current);
            eprintln!(" {} |{}{}", " ".repeat(line_loc_str.len()), " ".repeat(loc.start_column as usize), "^".repeat((loc.end_column - loc.start_column + 1)as usize));
            if let Some(line) = next_line {
                eprintln!(" {} |{}", " ".repeat(line_loc_str.len()), line);
            }
        }
    }

    pub fn print_error(&self, error: &dyn CompilationError) {
        eprintln!("--------------------");
        eprintln!("error: {}!", error.summarize());
        self.print_error_loc(error.loc());
        eprintln!(">{}", error.description())
    }
}

#[derive(Clone, Debug)]
pub enum TextProvider {
    Plain(String),
    File(PathBuf),
}

impl TextProvider {
    pub fn read_all(&self) -> String {
        match self {
            TextProvider::Plain(x) => x.clone(),
            TextProvider::File(x) => {
                let mut file = File::open(x).expect("Cannot open file");
                let mut str = String::new();
                file.read_to_string(&mut str).expect("Error reading file");
                str
            },
        }
    }
}
