use std::env;
use std::fs::File;
use std::io::Read;

pub fn main() {
    let file = match env::args().nth(1) {
        Some(x) => x,
        None => {
            eprintln!("Usage: clan <file>");
            return
        }
    };
    let mut file = File::open(file).expect("Cannot open file");
    let mut str = String::new();
    file.read_to_string(&mut str).expect("Error reading file");
    let res = coldpiler::run(str);

    match res {
        Ok(x) => {
            println!("Returned: {}", x);
        },
        Err(_) => {
            eprintln!("Error");
        },// Already printed
    }
}

