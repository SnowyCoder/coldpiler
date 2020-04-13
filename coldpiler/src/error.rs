use coldpiler_parser::loc::SpanLoc;
use crate::TokenizationError;
use coldpiler_parser::parser::ParsingError;
use crate::lang::ScannerTokenType;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ErrorLoc {
    NoLocation(),
    SingleLocation(SpanLoc),
    DoubleLocation(SpanLoc, SpanLoc),
}

pub trait CompilationError {
    fn error_type(&self) -> String;

    fn loc(&self) -> ErrorLoc;

    fn summarize(&self) -> String;

    fn description(&self) -> String;
}

impl CompilationError for TokenizationError {
    fn error_type(&self) -> String {
        "Token not recognized".to_owned()
    }

    fn loc(&self) -> ErrorLoc {
        ErrorLoc::SingleLocation(self.0.span)
    }

    fn summarize(&self) -> String {
        "Unexpected char".to_owned()
    }

    fn description(&self) -> String {
        "Cannot recognize any token".to_owned()
    }
}

fn display_token(token: Option<ScannerTokenType>) -> String {
    match token {
        Some(x) => format!("token {:?}", x),
        None => "end of line".to_string(),
    }
}

impl CompilationError for ParsingError<ScannerTokenType> {
    fn error_type(&self) -> String {
        "Unexpected token".to_string()
    }

    fn loc(&self) -> ErrorLoc {
        ErrorLoc::SingleLocation(self.token_loc)
    }

    fn summarize(&self) -> String {
        format!("Unexpected {}", &display_token(self.token))
    }

    fn description(&self) -> String {
        let mut res = format!("Unexpected {}, expected: ", &display_token(self.token));
        if self.expected.is_empty() {
            // Should never happen, if there's a point where all of the lookahead are wrong then
            // the table should reject the input sooner.
            res.push_str("Nothing");
        } else {
            res.push('[');
            for (i, x) in self.expected.iter().enumerate() {
                if i != 0 {
                    res.push_str(", ");
                }

                res.push_str(&match x {
                    Some(x) => format!("{:?}", x),
                    None => "None".to_owned(),
                })
            }
            res.push(']')
        }
        res
    }
}


