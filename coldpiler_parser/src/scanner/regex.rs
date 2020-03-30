use super::nfa::NFA;
use super::scanner::CustomTokenType;
use std::slice::Iter;
use crate::util::char_range_inclusive;

///! This crate contains the logic to convert from a regex to a fully functional NFA
///! There isn't any hard "algorithm" in here, just some boilerplate code and some good
///! conventions.

fn concatenate<T: CustomTokenType>(nfas: &[&NFA<T>]) -> NFA<T> {
    let mut res = NFA::new();
    //res.reserve_nodes(nfas.iter().map(|x| x.node_count()).sum());

    let first_nfa = nfas.get(0);
    let first_nfa = if let Some(x) = first_nfa {
        x
    } else {
        return res;
    };

    let _previous_start = res.add_nfa(first_nfa);

    for nfa in nfas.iter().skip(1) {
        let node_count = res.node_count() as u32;
        let current_begin = res.add_nfa(nfa);
        if node_count > 1 {
            let previous_end = node_count - 1;
            res.add_edge(previous_end, current_begin, None);
        }
    }

    res
}

fn or_catenate<T: CustomTokenType>(nfas: &[&NFA<T>], combine_end: bool) -> NFA<T> {
    //        -> nfa1 |
    // first |        -> last
    //        -> nfa2 |
    let real_nfas = nfas.iter()
        .enumerate()
        .filter(|(_index, nfa)| {
            nfa.node_count() > 0
        });
    let mut res = NFA::new();

    let first = res.add_node(None);
    let begins: Vec<(usize, u32)> = real_nfas.clone().map(|(index, nfa)| (index, res.add_nfa(nfa))).collect();

    let end = if combine_end {
        let index = res.add_node(None);
        if nfas.iter().any(|x| x.node_count() == 0) {
            res.add_edge(first, index, None);
        }
        index
    } else {
        std::u32::MAX
    };

    for (nfa_index, start_index) in begins.iter() {
        res.add_edge(first, *start_index, None);
        if combine_end {
            res.add_edge(*start_index + nfas[*nfa_index].node_count() as u32 - 1, end, None);
        }
    }
    
    res
}

fn for_repeat<T: CustomTokenType>(nfa: &NFA<T>, times: usize) -> NFA<T> {
    let mut array = Vec::with_capacity(times);
    array.resize(times, nfa);
    concatenate(array.as_slice())
}

fn star_repeat<T: CustomTokenType>(nfa: &mut NFA<T>) {
    nfa.add_edge(nfa.node_count() as u32 - 1, 0, None);
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Ord, PartialOrd)]
struct CounterToken(i32);

struct GroupParseData {
    nfa: NFA<CounterToken>,
    last_nfa: Option<NFA<CounterToken>>,
    next_token: i32,
}

impl GroupParseData {
    fn flush_last(&mut self) {
        if let Some(last) = self.last_nfa.take() {
            self.nfa = concatenate(&[&self.nfa, &last])
        }
    }

    fn allocate_token(&mut self) -> CounterToken {
        let curr = self.next_token;
        self.next_token += 1;
        CounterToken(curr)
    }

    fn transform_finalize<T: CustomTokenType>(self, token: Option<T>, is_last_star: bool) -> NFA<T> {
        // The parsing method colors each generated nfa with a sequential token id, when the group parsing is done
        // every token is removed expect from the last one that is replaced with the real token, this function does
        // exactly that, iterating every node and transforming its token

        // If the last character is a star there isn't just one target token
        // But two because two nfas could be correct (the starred element and
        // the one before him)
        let target_token = self.next_token - 1 + if is_last_star { -1 } else { 0 };

        let replaced_token = token;

        self.nfa.map_tokens(|t| t.and_then(|x| if x.0 >= target_token { replaced_token } else { None } ))
    }
}

fn split_par<'a>(report: &mut RegexReportWriter, text: &'a str, par_end: char) -> Result<(&'a str, &'a str), ()> {
    let mut splitniter = text.splitn(2, par_end);
    let subtext = splitniter.next().unwrap();
    let after_text = match splitniter.next() {
        Some(x) => x,
        None => {
            report.add_report(RegexReportLevel::Error, "Unclosed parenthesis".to_string(), 0, 0);
            return Err(());
        }
    };
    Ok((subtext, after_text))
}

fn parse_char<'a, T: CustomTokenType>(report: &mut RegexReportWriter, text: &'a str, token: Option<T>) -> Result<(&'a str, NFA<T>), ()> {
    let first_char = text.chars().next().expect("Cannot parse empty sequence");
    report.visit_char(first_char);

    if first_char == '\\' {// Special char
        let second_char = match text.chars().nth(1) {
            Some(x) => x,
            None => {
                report.add_report(RegexReportLevel::Error, "Unclosed special char (use \\\\ to type \\)".to_string(), 0, 0);
                return Ok(("", NFA::new()))
            }
        };
        report.visit_char(second_char);
        let mut nfas = Vec::new();
        match second_char {
            'a' => {
                // Alpha chars
                let iter = char_range_inclusive('a', 'z').chain(char_range_inclusive('A', 'Z'));
                for x in iter {
                    nfas.push(NFA::from_char(x, token));
                }
            }
            's' => {
                nfas.push(NFA::from_char(' ', token));
                nfas.push(NFA::from_char('\t', token));
                nfas.push(NFA::from_char('\r', token));
                nfas.push(NFA::from_char('\n', token));
            }
            'x' => {
                // Hex chars
                let iter =  char_range_inclusive('a', 'z')
                    .chain(char_range_inclusive('A', 'Z'))
                    .chain(char_range_inclusive('0', '9'));
                for c in iter {
                    nfas.push(NFA::from_char(c, token));
                }
            }
            't' => nfas.push(NFA::from_char('\t', token)),
            'r' => nfas.push(NFA::from_char('\r', token)),
            'n' => nfas.push(NFA::from_char('\n', token)),
            '\\' => nfas.push(NFA::from_char('\\', token)),
            _ => {
                report.add_report(RegexReportLevel::Error, "Invalid special char (use \\\\ to type \\)".to_string(), 0, 1)
            }
        }
        let nfa_refs: Vec<_> = nfas.iter().collect();
        return Ok((&text[2..], or_catenate(&nfa_refs, true)))
    } else {
        return Ok((&text[1..], NFA::from_char(first_char, token)))
    }
}

fn parse_char_group<T: CustomTokenType>(report: &mut RegexReportWriter, mut text: &str, token: Option<T>) -> Result<NFA<T>, ()> {
    let mut nfas = Vec::new();
    while let Some(first_char) = text.chars().next() {
        if first_char == '-' {
            report.visit_char('-');
            nfas.push(NFA::from_text(&['-' as u8], token));
            text = &text[1..];
            continue
        }

        if text.chars().nth(1) == Some('-') && text.chars().nth(2).is_some() {
            // RANGE
            let from_range = text.chars().nth(0).unwrap();
            let to_range = text.chars().nth(2).unwrap();

            report.visit_char(from_range);
            report.visit_char('-');
            report.visit_char(to_range);

            if (from_range as u32) >= (to_range as u32) {
                report.add_report(RegexReportLevel::Error, format!("Invalid range {}-{}", from_range, to_range), 0, 2);
                return Err(())
            }

            for x in char_range_inclusive(from_range, to_range) {
                nfas.push(NFA::from_char(x, token));
            }
            text = &text[3..]
        } else {
            let (new_text, nfa) = parse_char(report, text, token)?;
            text = new_text;
            nfas.push(nfa);
        }
    }
    let ptrs: Vec<&NFA<T>> = nfas.iter().collect();
    let res = or_catenate(&ptrs, true);
    Ok(res)
}

fn parse_group<T: CustomTokenType>(report: &mut RegexReportWriter, text: &str, token: Option<T>) -> Result<NFA<T>, ()> {
    let mut data = GroupParseData {
        nfa: NFA::new(),
        last_nfa: None,
        next_token: 0,
    };
    let mut is_last_char_star = false;
    let mut titer = text.chars();

    while let Some(ch) = titer.next() {
        report.visit_char(ch);

        if ch == '|' {
            data.flush_last();
            let final_nfa = data.transform_finalize(token, is_last_char_star);
            let next_nfa = parse_group(report, titer.as_str(), token)?;
            return Ok(or_catenate(&[&final_nfa, &next_nfa], true))
        }

        is_last_char_star = false;
        match ch {
            '(' => {
                data.flush_last();

                let (subtext, after_text) = split_par(report,titer.as_str(), ')')?;
                titer = after_text.chars();

                data.last_nfa = Some(parse_group(report, subtext, Some(data.allocate_token()))?);
                report.visit_char(')')
            },
            '[' => {
                data.flush_last();
                let (subtext, after_text) = split_par(report, titer.as_str(), ']')?;
                titer = after_text.chars();

                let token = data.allocate_token();
                let res = parse_char_group(report, subtext, Some(token))?;
                data.last_nfa = Some(res);
                report.visit_char(']')
            },
            '*' | '+' => {
                // + and * only differ in the token assignment
                let last_nfa = match data.last_nfa.as_mut() {
                    Some(x) => x,
                    None => {
                        report.add_report(RegexReportLevel::Error, format!("Invalid group for {}", ch), 0, 0);
                        // Continue, this is a "soft" error
                        continue
                    }
                };
                star_repeat(last_nfa);
                data.flush_last();
                is_last_char_star = ch == '*';
            }
            '{' => {
                let (subtext, after_text) = split_par(report, titer.as_str(), '}')?;
                titer = after_text.chars();

                match subtext.parse::<u32>() {
                    Ok(count) => {
                        match &data.last_nfa {
                            Some(last_nfa) => {
                                data.last_nfa = Some(for_repeat(last_nfa, count as usize));
                            },
                            None => {
                                report.add_report(RegexReportLevel::Error, "Invalid repeat".to_string(), 0, 0);
                            }
                        }
                    },
                    Err(_) => {
                        report.add_report(RegexReportLevel::Error, "Invalid int".to_string(), 1, subtext.len());
                        // continue (soft error)
                    },
                }
                report.advance_index(subtext.bytes().len());
                report.visit_char('}')
            }
            _ => {
                // Adding text
                data.flush_last();
                data.last_nfa = Some(NFA::from_text(ch.to_string().as_bytes(), Some(data.allocate_token())));
            }
        }
    };

    data.flush_last();
    Ok(data.transform_finalize(token, is_last_char_star))
}

pub fn regex_to_nfa<T: CustomTokenType>(report: &mut RegexReport, regex: &str, token: Option<T>) -> Result<NFA<T>, ()> {
    let mut rep_writer = report.writer();
    parse_group(&mut rep_writer, regex, token)
}

pub fn regex_map_to_nfa<T>(report: &mut RegexReport, regex_map: &[(T, &str)]) -> Result<NFA<T>, ()>
    where T: CustomTokenType {

    let nfas: Vec<Result<NFA<T>, ()>> = regex_map.iter()
        .enumerate()
        .map(|(index, &(token, regex))| {
            let mut reporter = report.writer();
            reporter.set_default_entry(index);
            parse_group(&mut reporter, regex, Some(token))
    }).collect();
    let nfa_refs: Result<Vec<&NFA<T>>, &()> = nfas.iter().map(|x| x.as_ref()).collect();
    let nfa_refs = nfa_refs.map_err(|x| *x)?;
    Ok(or_catenate(nfa_refs.as_slice(), false))
}

#[derive(Debug, Eq, PartialEq, Default)]
pub struct RegexReport {
    reports: Vec<RegexReportEntry>,
}

pub struct RegexReportWriter<'a> {
    report: &'a mut RegexReport,
    default_entry: usize,
    // The current info is written like this:
    // abcd...h...tuvz
    //       ^^ current_char_bytes
    //       | current_char_index
    // this awful lineart just to say that the current_char_index doesn't include the current char
    // count of bytes
    current_index: usize,// parsed bytes count
    current_char_bytes: u8,// current visited char bytes count
}

impl RegexReport {
    pub fn new() -> Self {
        RegexReport {
            reports: vec![],
        }
    }

    pub fn writer(&mut self) -> RegexReportWriter {
        RegexReportWriter {
            report: self,
            default_entry: 0,
            current_index: 0,
            current_char_bytes: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.iter().next().is_none()
    }

    pub fn iter(&self) -> Iter<RegexReportEntry> {
        self.reports.iter()
    }
}

impl RegexReportWriter<'_> {
    pub fn add_report(&mut self, level: RegexReportLevel, description: String, start: usize, end: usize) {
        self.report.reports.push(RegexReportEntry {
            level,
            description,
            regex_entry: self.default_entry,
            start_index: start + self.current_index,
            end_index: end + self.current_index,
        })
    }

    pub fn set_default_entry(&mut self, entry: usize) {
        self.default_entry = entry;
    }

    pub fn advance_index(&mut self, ch_num: usize) {
        self.current_index += self.current_char_bytes as usize + ch_num;
        self.current_char_bytes = 0;
    }

    pub fn visit_char(&mut self, ch: char) {
        self.current_index += self.current_char_bytes as usize;
        self.current_char_bytes = ch.len_utf8() as u8;
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum RegexReportLevel {
    Warning, Error,
}

#[derive(Debug, PartialEq, Eq)]
pub struct RegexReportEntry {
    pub level: RegexReportLevel,
    pub description: String,
    pub regex_entry: usize,// The entry index where the error was thrown (0 if the context doesn't make sense)
    pub start_index: usize,
    pub end_index: usize,
}


#[cfg(test)]
mod tests {
    use crate::scanner::*;
    use crate::scanner::regex::{RegexReport, RegexReportEntry, RegexReportLevel};

    #[test]
    fn test_simple() {
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
        enum TestTokenType {
            End
        }

        let mut report = RegexReport::new();

        let dfa = regex_to_nfa(&mut report, "ad(b|c)*", Some(TestTokenType::End));
        assert!(report.is_empty());
        let dfa = dfa.unwrap().to_dfa();

        assert_eq!(None, dfa.run_single_token("adba", 0));
        assert_eq!(Some(TestTokenType::End), dfa.run_single_token("adbcbcbc", 0));
        assert_eq!(None, dfa.run_single_token("a", 0));
        assert_eq!(Some(TestTokenType::End), dfa.run_single_token("ad", 0));
        assert_eq!(None, dfa.run_single_token("adbcca", 0));
    }

    #[test]
    fn test_raw_chars() {
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        enum TestTokenType {
            End
        }
        let mut report = RegexReport::new();
        let dfa = regex_to_nfa(&mut report, "[ab]", Some(TestTokenType::End));
        assert!(report.is_empty());
        let dfa = dfa.unwrap().to_dfa().minimize_hopcroft();

        assert_eq!(None, dfa.run_single_token("c", 0));
        assert_eq!(Some(TestTokenType::End), dfa.run_single_token("a", 0));
        assert_eq!(None, dfa.run_single_token("ab", 0));
        assert_eq!(Some(TestTokenType::End), dfa.run_single_token("b", 0));
        assert_eq!(None, dfa.run_single_token("ca", 0));
    }

    #[test]
    fn test_map_simple() {
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
        enum TestTokenType {
           A, B
        }
        use TestTokenType::*;

        let mut report = RegexReport::new();

        let _nfa = regex_map_to_nfa(&mut report, &[
            (A, "b"),
            (B, "a"),
        ]);
        assert!(report.is_empty())
    }

    #[test]
    fn test_map() {
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        enum TestTokenType {
            Space, Not, Nor, And, BinaryNumber
        }
        use TestTokenType::*;

        let mut report = RegexReport::new();

        let nfa = regex_map_to_nfa(&mut report, &[
            (Space, "[ \t\n]+"),
            (Not, "not"),
            (Nor, "nor"),
            (And, "and"),
            (BinaryNumber, "[0123456789]+"),
        ]);
        assert!(report.is_empty());
        let dfa = nfa.unwrap().to_dfa().minimize_hopcroft();

        assert_eq!(dfa.tokenize("not  \n0 nor 15 and", 0), [
            Token { text: "not".to_string(), ttype: TokenType::Custom(Not) },
            Token { text: "  \n".to_string(), ttype: TokenType::Custom(Space) },
            Token { text: "0".to_string(), ttype: TokenType::Custom(BinaryNumber) },
            Token { text: " ".to_string(), ttype: TokenType::Custom(Space) },
            Token { text: "nor".to_string(), ttype: TokenType::Custom(Nor) },
            Token { text: " ".to_string(), ttype: TokenType::Custom(Space) },
            Token { text: "15".to_string(), ttype: TokenType::Custom(BinaryNumber) },
            Token { text: " ".to_string(), ttype: TokenType::Custom(Space) },
            Token { text: "and".to_string(), ttype: TokenType::Custom(And) },
        ]);
    }

    #[test]
    fn test_misc() {
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        enum TestTokenType {
            A, B,
        }
        use TestTokenType::*;

        let mut report = RegexReport::new();

        let nfa = regex_map_to_nfa(&mut report, &[
            (A, "[0123456789]+"),
            (B, r"[\s]+"),
        ]);
        assert!(report.is_empty());
        let dfa = nfa.unwrap().to_dfa();

        assert_eq!(TokenType::Custom(TestTokenType::B), dfa.tokenize(" \t  \n\t", 0)[0].ttype);

        // Test some obscure bug in hopcroft's split algorithm
        let nfa = regex_map_to_nfa(&mut report, &[
            (A, "var"),
            (B, r"[av][ar]*|k"),
        ]);
        assert!(report.is_empty());
        let dfa = nfa.unwrap().to_dfa();

        assert_eq!(TokenType::Custom(TestTokenType::A), dfa.tokenize("var", 0)[0].ttype);
        assert_eq!(TokenType::Custom(TestTokenType::B), dfa.tokenize("vara", 0)[0].ttype);
        assert_eq!(TokenType::Custom(TestTokenType::B), dfa.tokenize("k", 0)[0].ttype);

        // Other tests
        let nfa = regex_map_to_nfa(&mut report, &[
            (A, r"aa*|b"),
        ]);
        let nfa = nfa.unwrap();
        assert!(report.is_empty());
        eprintln!("NFA: {}", nfa);
        let dfa = nfa.to_dfa();


        //assert_eq!(TokenType::Custom(TestTokenType::A), dfa.tokenize("var", 0)[0].ttype);
        assert_eq!(TokenType::Custom(TestTokenType::A), dfa.tokenize("a", 0)[0].ttype);
    }

    #[test]
    fn test_errors() {
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        enum TestTokenType {
            A, B,
        }
        use TestTokenType::*;
        let mut report = RegexReport::new();

        let nfa = regex_to_nfa(&mut report, "+a[ab](b", Some(A));
        assert_eq!(
            RegexReport {
                reports: vec![
                    RegexReportEntry {
                        level: RegexReportLevel::Error,
                        description: "Invalid group for +".to_string(),
                        regex_entry: 0, start_index: 0, end_index: 0
                    },
                    RegexReportEntry {
                        level: RegexReportLevel::Error,
                        description: "Unclosed parenthesis".to_string(),
                        regex_entry: 0, start_index: 6, end_index: 6
                    }
                ],
            },
            report
        );
        assert!(nfa.is_err());

        let mut report = RegexReport::new();
        let nfa = regex_map_to_nfa(&mut report,
        &[
            (A, "a{aaa}"),
            (B, "{5}b"),
        ]);
        assert_eq!(RegexReport {
            reports: vec![
                RegexReportEntry {
                    level: RegexReportLevel::Error,
                    description: "Invalid int".to_string(),
                    regex_entry: 0, start_index: 2, end_index: 4
                },
                RegexReportEntry {
                    level: RegexReportLevel::Error,
                    description: "Invalid repeat".to_string(),
                    regex_entry: 1, start_index: 0, end_index: 0
                }
            ]
        }, report);
        assert!(nfa.is_ok());
    }
}
