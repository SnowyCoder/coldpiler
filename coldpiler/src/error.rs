use coldpiler_parser::loc::SpanLoc;

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


