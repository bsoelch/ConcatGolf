use std::fs;
use std::io::{self, Write,stdout};
use std::collections::HashMap;
use std::sync::OnceLock;
use std::ops::Range;
use std::vec::IntoIter;
use std::str;
use std::slice;

// tokenizer
#[derive(Debug,PartialEq,Clone,Copy)]
enum TokenType{
    Word,
    CharString,
    ByteString,
    OpenBracket,
    ClosingBracket,
    EOF
}
impl ToString for TokenType {
    fn to_string(&self) -> String {
        match self {
            TokenType::Word => "Word",
            TokenType::CharString => "CharString",
            TokenType::ByteString => "ByteString",
            TokenType::OpenBracket => "OpenBracket",
            TokenType::ClosingBracket => "ClosingBracket",
            TokenType::EOF => "EOF",
        }.to_string()
    }
}
const BRACKET_CHARS: &[u8] = b"{}";
fn is_special_char(op_char: u8) -> bool {
    BRACKET_CHARS.contains(&op_char)
}
fn is_open_bracket_char(op_char: u8) -> bool {
    op_char==b'{'
}
#[derive(Debug,Clone,Copy)]
struct Position{
    line :i32,
    line_pos: i32
}
#[derive(Debug,Clone,Copy)]
struct Token<'a>{
    token_type: TokenType,
    value: &'a [u8],
    pos: Position
}

impl ToString for Token<'_> {
    fn to_string(&self) -> String {
        format!("{:?}: \"{}\" at {}:{}", self.token_type, String::from_utf8_lossy(self.value), self.pos.line, self.pos.line_pos)
    }
}

fn tokenize<'a>(input: &'a [u8]) -> Vec<Token<'a>> {
    let mut start_index: usize=0;
    let mut line = 1;
    let mut line_pos = 0;
    let mut start_line = 1;
    let mut start_pos = 0;
    let mut output = Vec::new();
    let mut string_start = 0;
    let mut escaped = false;
    for (i, &c) in input.into_iter().enumerate() {
        if c == b'\n' {
            line_pos = 0;
            line += 1;
        } else {
            line_pos += 1;
        }
        if escaped {
            escaped = false;
            continue
        }
        if string_start != 0 {
            if c == b'\\' {
                escaped = true;
            }
            if c != string_start {
                continue
            }
        }
        let is_operator = is_special_char(c);
        if c <= 0x20 /*ascii control-character or space*/ || is_operator || c == b'"' || c == b'\'' {
            if string_start != 0 || start_index < i {
                let token_value = &input[start_index..i];
                output.push(Token{
                    token_type: if string_start == b'\'' {
                        TokenType::CharString
                    } else if string_start == b'"' {
                        TokenType::ByteString
                    } else {
                        TokenType::Word
                    },
                    value: token_value,
                    pos: Position{line:start_line,line_pos: start_pos}
                });
            }
            if string_start != 0 {
                string_start = 0;
            } else if c == b'"' || c == b'\'' {
                string_start = c;
            }
            if is_operator {
                output.push(Token{
                    token_type: if is_open_bracket_char(c) {TokenType::OpenBracket} else {TokenType::ClosingBracket},
                    value: &input[i..(i+1)],
                    pos: Position{line:line,line_pos: line_pos-1}
                });
            }
            start_index= i+1;
            start_line = line;
            start_pos = line_pos;
        }
    }
    // ensure last token is finished
    let token_value = &input[start_index..];
    if token_value.len() > 0 {
        let first_char = token_value[0];
        let token_type = if is_open_bracket_char(first_char) {
            TokenType::OpenBracket
        } else if is_special_char(first_char) {
            TokenType::ClosingBracket
        } else if string_start == b'\'' {
            TokenType::CharString
        } else if string_start == b'"' {
            TokenType::ByteString
        } else {
            TokenType::Word
        };
        output.push(Token{
            token_type: token_type,
            value: token_value,
            pos: Position{line:start_line,line_pos: start_pos}
        });
    }
    output.push(Token{
        token_type: TokenType::EOF,
        value: b"",
        pos: Position{line:line,line_pos: line_pos}
    });
    return output
}

// parser
#[allow(non_camel_case_types)]
#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash)]
enum BuiltIn {
    DUP, DUP2,
    OVER, OVER2, OVER_N, UNDER,
    SWAP, SWAP2, ROT2, ROT_2, ROT3, ROT_3, ROT_N,
    DROP, DROP2, DROP_N,
    // TODO? more stack manipulators ? 'nip' (discard elements below top value)

    NEGATE,
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    MODULO,
    POW,

    AND,
    OR,
    XOR,
    NOT,

    LEN,
    COLLECT,
    COLLECT1,
    COLLECT2,
    SPLAT,

    IF,
    REPEAT,
    FOR,
    MAP,

    CALL,
}
const BUILTIN_WORDS: [(&[u8],BuiltIn); 51] = [
    // stack manipulation
    (b"dup", BuiltIn::DUP),
    (b"dup2", BuiltIn::DUP2),
    (b"over", BuiltIn::OVER),
    (b"over2", BuiltIn::OVER2),
    (b"over3", BuiltIn::OVER2),
    (b"over*", BuiltIn::OVER_N),
    (b"under", BuiltIn::UNDER),
    (b"swap", BuiltIn::SWAP),
    (b"swap2", BuiltIn::SWAP2),
    (b"rot2", BuiltIn::ROT2),
    (b"rot-2", BuiltIn::ROT_2),
    (b"rot3", BuiltIn::ROT3),
    (b"rot-3", BuiltIn::ROT_3),
    (b"rot*", BuiltIn::ROT_N),
    (b"drop", BuiltIn::DROP),
    (b"drop2", BuiltIn::DROP2),
    (b"drop*", BuiltIn::DROP_N),
    // arithmetic operators
    (b"-_", BuiltIn::NEGATE),
    (b"negate", BuiltIn::NEGATE),
    (b"+", BuiltIn::ADD),
    (b"add", BuiltIn::ADD),
    (b"-", BuiltIn::SUBTRACT),
    (b"sub", BuiltIn::SUBTRACT),
    (b"*", BuiltIn::MULTIPLY),
    (b"mul", BuiltIn::MULTIPLY),
    (b"/", BuiltIn::DIVIDE),
    (b"div", BuiltIn::DIVIDE),
    (b"%", BuiltIn::MODULO),
    (b"mod", BuiltIn::MODULO),
    (b"^", BuiltIn::POW),
    (b"pow", BuiltIn::POW),
    // logic operators
    (b"&", BuiltIn::AND),
    (b"and", BuiltIn::AND),
    (b"|", BuiltIn::OR),
    (b"or", BuiltIn::OR),
    (b"xor", BuiltIn::XOR),
    (b"!", BuiltIn::NOT),
    (b"not", BuiltIn::NOT),
    // lists
    (b"len", BuiltIn::LEN),
    (b"collect", BuiltIn::COLLECT),
    (b"collect1", BuiltIn::COLLECT1),
    (b"collect2", BuiltIn::COLLECT2),
    (b"...", BuiltIn::SPLAT),
    (b"splat", BuiltIn::SPLAT),
    // control flow
    (b"?", BuiltIn::IF),
    (b"if", BuiltIn::IF),
    (b"repeat", BuiltIn::REPEAT),
    (b"for", BuiltIn::FOR),
    (b"map", BuiltIn::MAP),
    // functions
    (b"()", BuiltIn::CALL),
    (b"call", BuiltIn::CALL),
];
static BUILTIN_INFO: OnceLock<HashMap<&[u8],BuiltIn>> = OnceLock::new();
static BUILTIN_NAMES: OnceLock<HashMap<BuiltIn,&[u8]>> = OnceLock::new();
fn builtin_id(word: &[u8]) -> Option<BuiltIn> {
    BUILTIN_INFO.get_or_init(|| {
        let mut map = HashMap::new();
        for (key, value) in BUILTIN_WORDS {
            map.insert(key, value);
        }
        map
    }).get(word).map(|v|*v)
}
fn builtin_name(built_in: &BuiltIn) -> Option<&'static [u8]> {
    BUILTIN_NAMES.get_or_init(|| {
        let mut map = HashMap::new();
        for (name, id) in BUILTIN_WORDS {
            map.insert(id, name);
        }
        map
    }).get(built_in).map(|v|*v)
}

struct EscapedByteIter<'a> {
    byte_iter: slice::Iter<'a,u8>,
    buffer: [u8; 7],
    buff_size: u8
}
impl<'a> EscapedByteIter<'a> {
    fn buff_add(&mut self,val: u8) {
        if self.buff_size >= 7 {
            panic!("buffer overflow");
        }
        self.buffer[self.buff_size as usize] = val;
        self.buff_size += 1;
    }
    fn buff_get(&mut self) -> u8 {
        if self.buff_size == 0 {
            panic!("buffer underflow");
        }
        let res = self.buffer[0];
        self.buff_size -= 1;
        for i in 0..=5 {
            self.buffer[i] = self.buffer[i+1];
        }
        self.buffer[6] = 0;
        return res;
    }
    fn get_hex_char(&mut self) -> u32 {
        if self.buff_size > 0 {
            return 0
        }
        let next_byte = self.byte_iter.next().copied();
        if next_byte.is_none() {
            return 0
        }
        let byte_val = next_byte.unwrap();
        if byte_val >= b'0' && byte_val <= b'9' {
            return  (byte_val - b'0') as u32;
        }
        if byte_val >= b'A' && byte_val <= b'F' {
            return  (byte_val - b'A') as u32 + 10;
        }
        if byte_val >= b'a' && byte_val <= b'f' {
            return  (byte_val - b'a') as u32 + 10;
        }
        self.buff_add(byte_val);
        return 0
    }
    fn encode_utf8(&mut self, codepoint: u32) -> u8 {
        if codepoint < 0x80 {
            return codepoint as u8
        }
        let next_val = if self.buff_size > 1 {
            panic!("assertion failed: buffer should contain at most one element");
        } else if self.buff_size == 1 {
            Some(self.buff_get())
        } else {
            None
        };
        if codepoint < 0x800 { // 0x110*** ** 0x10** ****
            self.buff_add((codepoint&0x3f) as u8 | 0x80);
            if next_val.is_some() {self.buff_add(next_val.unwrap())}
            return (codepoint>>6) as u8 | 0xC0
        } else if codepoint < 0x10000 { // 0x1110**** 0x10 **** ** 0x10** ****
            self.buff_add(((codepoint>>6)&0x3f) as u8 | 0x80);
            self.buff_add((codepoint&0x3f) as u8 | 0x80);
            if next_val.is_some() {self.buff_add(next_val.unwrap())}
            return (codepoint>>12) as u8 | 0xE0
        } else if codepoint < 0x200000 { // 0x11110* ** 0x10 ** **** 0x10**** ** 0x10** ****
            self.buff_add(((codepoint>>12)&0x3f) as u8 | 0x80);
            self.buff_add(((codepoint>>6)&0x3f) as u8 | 0x80);
            self.buff_add((codepoint&0x3f) as u8 | 0x80);
            if next_val.is_some() {self.buff_add(next_val.unwrap())}
            return (codepoint>>18) as u8 | 0xF0
        } else if codepoint < 0x4000000 { // 0x111110** 0x10**** ** 0x10 ** **** 0x10**** ** 0x10** ****
            self.buff_add(((codepoint>>18)&0x3f) as u8 | 0x80);
            self.buff_add(((codepoint>>12)&0x3f) as u8 | 0x80);
            self.buff_add(((codepoint>>6)&0x3f) as u8 | 0x80);
            self.buff_add((codepoint&0x3f) as u8 | 0x80);
            if next_val.is_some() {self.buff_add(next_val.unwrap())}
            return (codepoint>>24) as u8 | 0xF8
        // 0x1111110* 0x10** **** 0x10**** ** 0x10 ** **** 0x10**** ** 0x10** ****
        } else if codepoint < 0x80000000 {
            self.buff_add(((codepoint>>24)&0x3f) as u8 | 0x80);
            self.buff_add(((codepoint>>18)&0x3f) as u8 | 0x80);
            self.buff_add(((codepoint>>12)&0x3f) as u8 | 0x80);
            self.buff_add(((codepoint>>6)&0x3f) as u8 | 0x80);
            self.buff_add((codepoint&0x3f) as u8 | 0x80);
            if next_val.is_some() {self.buff_add(next_val.unwrap())}
            return (codepoint>>30) as u8 | 0xFC
        } else {
            self.buff_add(((codepoint>>30)&0x3f) as u8 | 0x80);
            self.buff_add(((codepoint>>24)&0x3f) as u8 | 0x80);
            self.buff_add(((codepoint>>18)&0x3f) as u8 | 0x80);
            self.buff_add(((codepoint>>12)&0x3f) as u8 | 0x80);
            self.buff_add(((codepoint>>6)&0x3f) as u8 | 0x80);
            self.buff_add((codepoint&0x3f) as u8 | 0x80);
            if next_val.is_some() {self.buff_add(next_val.unwrap())}
            return 0xFE
        }
    }
}
fn escaped_byte_iter<'a>(val: &'a [u8]) -> EscapedByteIter<'a> {
    EscapedByteIter{byte_iter: val.into_iter(),buffer:[0,0,0,0,0,0,0],buff_size:0}
}
impl<'a> Iterator for EscapedByteIter<'a> {
    type Item = u8;
    fn next(&mut self) -> Option<Self::Item> {
        let next = if self.buff_size > 0 {
            Some(self.buff_get())
        } else {
            self.byte_iter.next().copied()
        };
        if next.is_none() || next.unwrap() != b'\\' { // escape-esquence -> go to next byte
            return next;
        }
        let escaped = self.byte_iter.next().copied();
        if escaped.is_none() {
            return None
        }
        match escaped.unwrap() {
            b'n' => return Some(b'\n'),
            b't' => return Some(b'\t'),
            b'r' => return Some(b'\r'),
            b'x' => {
                let byte_val = (16 * self.get_hex_char()) + self.get_hex_char();
                return Some(byte_val as u8);
            },
            b'u' => {
                let mut codepoint = 0;
                for _ in 0..4 {
                    codepoint = 16* codepoint + self.get_hex_char();
                }
                return Some(self.encode_utf8(codepoint));
            },
            b'U' => {
                let mut codepoint = 0;
                for _ in 0..8 {
                    codepoint = 16* codepoint + self.get_hex_char();
                }
                return Some(self.encode_utf8(codepoint));
            },
            _ => return escaped
        }
    }
    fn size_hint(&self) -> (usize,Option<usize>) {
        let byte_size = self.byte_iter.size_hint();
        // worst case: sequence of '\U00000000' will have length 1/10 of original bytes
        ((byte_size.0+9)/10,byte_size.1)
    }
}
struct EscapedCharIter<'a> {
    byte_iter: EscapedByteIter<'a>
}
// decoder for a string with escape sequences using an extension of utf8
// * treat bytes between 0x80 and 0xBF as codepoints when they appear outside multi-byte sequences
// * pad mult-byte sequences to correct length (by appending 0x80)
// * allow codepoints up to 0x3ffffffffff
fn escaped_char_iter<'a>(val: &'a [u8]) -> EscapedCharIter<'a> {
    EscapedCharIter{byte_iter: escaped_byte_iter(val)}
}
impl<'a> EscapedCharIter<'a>{
    fn update_value(&mut self,mut value: u64) -> u64 {
        value = value << 6;
        if self.byte_iter.buff_size > 0 {
            return value;
        }
        let next = self.byte_iter.next();
        if next.is_none() || next.unwrap() < 0x80 || next.unwrap() >= 0xC0 {
            self.byte_iter.buff_add(next.unwrap());
            return value;
        }
        return value | (next.unwrap() & 0x3f) as u64;
    }
}
impl<'a> Iterator for EscapedCharIter<'a> {
    type Item = u64;
    fn next(&mut self) -> Option<Self::Item> {
        let next = self.byte_iter.next();
        if next.is_none() {
            return None;
        }
        let mut value = next.unwrap() as u64;
        if value < 0xC0 { // not the start of an utf8 sequence
            return Some(value)
        } else if value < 0xE0 { // 110* ****
            value = value & 0x1F;
            value = self.update_value(value);
            return Some(value)
        } else if value < 0xF0 { // 1110 ****
            value = value & 0x0F;
            value = self.update_value(value);
            value = self.update_value(value);
            return Some(value)
        } else if value < 0xF8 { // 1111 0***
            value = value & 0x07;
            value = self.update_value(value);
            value = self.update_value(value);
            value = self.update_value(value);
            return Some(value)
        } else if value < 0xFC { // 1111 10**
            value = value & 0x03;
            value = self.update_value(value);
            value = self.update_value(value);
            value = self.update_value(value);
            value = self.update_value(value);
            return Some(value)
        } else if value < 0xFE { // 1111 110*
            value = value & 0x01;
            value = self.update_value(value);
            value = self.update_value(value);
            value = self.update_value(value);
            value = self.update_value(value);
            value = self.update_value(value);
            return Some(value)
        } else if value == 0xFE { // 1111 1110
            value = 0;
            value = self.update_value(value);
            value = self.update_value(value);
            value = self.update_value(value);
            value = self.update_value(value);
            value = self.update_value(value);
            value = self.update_value(value);
            return Some(value)
        } else { // 1111 1111
            value = 0;
            value = self.update_value(value);
            value = self.update_value(value);
            value = self.update_value(value);
            value = self.update_value(value);
            value = self.update_value(value);
            value = self.update_value(value);
            value = self.update_value(value);
            return Some(value)
        }
    }
}


#[derive(Debug,Clone)]
enum Atom<'a> {
    BuiltInWord(BuiltIn),
    Number(f64), // TODO? use arbitary pression library for numbers (? rug crate )
    Assign(&'a [u8]),
    Get(&'a [u8]),
    ByteString(&'a [u8]),
    CharString(&'a [u8]),
    Quotation(Vec<Atom<'a>>),
}
fn dump_atoms<'a>(out_file: &mut fs::File, atoms: &'a [Atom<'a>],indent: usize)-> io::Result<()> {
    for atom in atoms {
        match atom {
            Atom::BuiltInWord(built_in) => {
                writeln!(out_file,"{}BuiltIn({})","   ".repeat(indent),String::from_utf8_lossy(builtin_name(built_in).unwrap()))?;
            }
            Atom::Number(value) => {
                writeln!(out_file,"{}Number({})","   ".repeat(indent),value)?;
            }
            Atom::ByteString(value) => { // TODO! escape special characters in string values
                writeln!(out_file,"{}ByteString({})","   ".repeat(indent),String::from_utf8_lossy(value))?;
            }
            Atom::CharString(value) => {
                writeln!(out_file,"{}CharString({})","   ".repeat(indent),String::from_utf8_lossy(value))?;
            }
            Atom::Quotation(elts) => {
                writeln!(out_file,"{}{{","   ".repeat(indent))?;
                dump_atoms(out_file,&elts,indent+1)?;
                writeln!(out_file,"{}}}","   ".repeat(indent))?;
            }
            Atom::Assign(name) => {
                writeln!(out_file,"{}Assign({})","   ".repeat(indent),String::from_utf8_lossy(name))?;
            }
            Atom::Get(name) => {
                writeln!(out_file,"{}Get({})","   ".repeat(indent),String::from_utf8_lossy(name))?;
            }
        }
    }
    Ok(())
}

fn try_parse_atom<'a>(mut tokens: &'a [Token<'a>]) -> Result<(Atom<'a>,usize),&'a Token<'a>> {
    match tokens[0].token_type {
        TokenType::Word => {
            let word = tokens[0].value;
            if word.len() > 0 && word[0].is_ascii_digit() ||
                word.len() > 1 && word[0] == b'-' && word[1].is_ascii_digit() {
                if let Ok(str_val) = str::from_utf8(word) {
                    if let Ok(value) = str_val.parse::<f64>() {
                        return Ok((Atom::Number(value),1))
                    }
                }
                return Err(&tokens[0])
            } else if word.len() > 0 && word.ends_with(b"=") {
                return Ok((Atom::Assign(&word[..(word.len()-1)]),1))
            } else if let Some(built_in) = builtin_id(word) {
                return Ok((Atom::BuiltInWord(built_in),1))
            }
            return Ok((Atom::Get(word),1))
        },
        TokenType::CharString => return Ok((Atom::CharString(tokens[0].value),1)),
        TokenType::ByteString => return Ok((Atom::ByteString(tokens[0].value),1)),
        TokenType::OpenBracket => {
            let mut consumed = 1;
            tokens=&tokens[1..];
            let mut children: Vec<Atom> = Vec::new();
            while tokens.len() > 0 {
                if tokens[0].token_type == TokenType::ClosingBracket {
                    consumed+=1;
                    break // reached end of quotation
                }
                let (atom,k) = try_parse_atom(tokens)?;
                children.push(atom);
                consumed+=k;
                tokens=&tokens[k..];
            }
            return Ok((Atom::Quotation(children),consumed));
        },
        _ => return Err(&tokens[0]),
    }
}
fn parse_program<'a>(mut tokens: &'a [Token<'a>]) -> Vec<Atom<'a>> {
    let mut children: Vec<Atom> = Vec::new();
    while tokens.len() > 0 {
        if tokens[0].token_type == TokenType::EOF {
            break // reached end of file
        }
        match try_parse_atom(tokens) {
            Ok((atom,k)) => {
                children.push(atom);
                tokens=&tokens[k..];
            },
            Err(token) => {
                println!("Unexpected token: {:?}",token);
                break
            }
        }
    }
    children
}

// interperter
#[derive(Debug,Clone)]
enum Value<'a>{
    Number(f64),
    ByteString(&'a [u8]),
    List(Vec<Value<'a>>),
    Quotation(&'a[Atom<'a>]),
}
impl Default for Value<'_> {
    fn default() -> Self {
        Value::Number(0f64)
    }
}
impl ToString for Value<'_> {
    fn to_string(&self) -> String {
        match self {
            Value::Number(num) => {
                return format!("Number({})",num);
            }
            Value::ByteString(val) => {
                return format!("ByteString({})", String::from_utf8_lossy(val));
            }
            Value::List(elts) => {
                let mut res = String::from("List: [");
                let mut first = true;
                for elt in elts {
                    if !first {
                        res+=", ";
                    }
                    first = false;
                    res += &elt.to_string();
                }
                return res+"]";
            }
            // TODO? print (part of) body
            Value::Quotation(_code) => String::from("Quotation: {...}")
        }
    }
}
fn unary_number_op0<'a>(val: Value<'a>,f: &dyn Fn(f64)->Value<'a>) -> Value<'a> {
    match val {
        Value::Number(num) => f(num),
        Value::Quotation(_) => {panic!("unsuported operand for unary operation");}
        _ => return Value::List(as_iter(val).map(|v|unary_number_op0(v,f)).collect())
    }
}
fn unary_number_op<'a>(stack: &mut Vec<Value<'a>>,f: fn(f64)->Value<'a>) {
    let val = stack.pop().unwrap_or_default();
    stack.push(unary_number_op0(val,&f));
}
fn binary_number_op0<'a>(left: Value<'a>,right: Value<'a>,f: fn(f64,f64)->Value<'a>)->Value<'a> {
    match right {
        Value::Number(right_num) => {
            return unary_number_op0(left,&|lhs|f(lhs,right_num));
        }
        Value::Quotation(_body) => {panic!("unsuported operands for binary operation");}
        _ => {}
    }
    match left {
        Value::Number(left_num) => {
            return unary_number_op0(right,&|rhs|f(left_num,rhs));
        }
        Value::Quotation(_body) => {panic!("unsuported operands for binary operation");}
        _ => {}
    }
    let mut left_iter = as_iter(left);
    let mut right_iter = as_iter(right);
    let mut res = Vec::with_capacity(std::cmp::max(left_iter.size_hint().0,right_iter.size_hint().0));
    loop {
      let lhs = left_iter.next();
      let rhs = right_iter.next();
      if lhs.is_none() && rhs.is_none() { break; }
      res.push(binary_number_op0(lhs.unwrap_or_default(),rhs.unwrap_or_default(),f));
    }
    return Value::List(res);
}
fn binary_number_op<'a>(stack: &mut Vec<Value<'a>>,f: fn(f64,f64)->Value<'a>) {
    let right = stack.pop().unwrap_or_default();
    let left = stack.pop().unwrap_or_default();
    stack.push(binary_number_op0(left,right,f));
}

fn eval_call<'a>(val: &Value<'a>,stack: &mut Vec<Value<'a>>,globals: &mut HashMap<&'a [u8],Value<'a>>) {
    match val {
        Value::Number(_num) => {panic!("cannot call number");} // ? create list with given number of elements
        Value::ByteString(_val) => {panic!("cannot call string");}
        Value::List(_elts) => {panic!("cannot call list");} // TODO? call for each element (with current stack)
        Value::Quotation(body) => {
            eval_block(body,stack,globals)
        }
    }
}
fn eval_or_value<'a>(val: &Value<'a>,stack: &mut Vec<Value<'a>>,globals: &mut HashMap<&'a [u8],Value<'a>>) {
    match val {
        Value::Quotation(body) => {
            eval_block(body,stack,globals)
        }
        _ => {stack.push(val.clone());}
    }
}
fn as_bool(val: Value) -> bool {
    return match val {
        Value::Number(num) => num != 0.0,
        Value::ByteString(val) => val.len() != 0,
        Value::List(elts) => elts.len() != 0,
        Value::Quotation(body) => body.len() != 0
    }
}

enum ValueIterator<'a> {
    Number(Range<i64>),
    NumberRev(std::iter::Rev<Range<i64>>),
    ByteString(EscapedByteIter<'a>),
    List(IntoIter<Value<'a>>),
    Quotation(slice::Iter<'a, Atom<'a>>),
}
impl<'a> Iterator for ValueIterator<'a> {
   type Item = Value<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ValueIterator::Number(range) => range.next().map(|i| Value::Number(i as f64)),
            ValueIterator::NumberRev(range) => range.next().map(|i| Value::Number(i as f64)),
            ValueIterator::ByteString(iter) => iter.next().map(|b|Value::Number(b as f64)),
            ValueIterator::List(iter) => iter.next(),
            ValueIterator::Quotation(iter) => iter.next().map(|x|Value::Quotation(slice::from_ref(x))),
        }
    }
    fn size_hint(&self) -> (usize,Option<usize>) {
        match self {
            ValueIterator::Number(range) => range.size_hint(),
            ValueIterator::NumberRev(range) => range.size_hint(),
            ValueIterator::ByteString(iter) => iter.size_hint(),
            ValueIterator::List(iter) => iter.size_hint(),
            ValueIterator::Quotation(iter) => iter.size_hint(),
        }
    }
}
fn as_iter(val : Value) -> ValueIterator {
    match val {
        Value::Number(num) => {
            if num >= 0.0 {
                return ValueIterator::Number(0..(num.ceil() as i64))
            } else {
                return ValueIterator::NumberRev((1..(1-num.ceil() as i64)).rev())
            }
        },
        Value::ByteString(msg) => return ValueIterator::ByteString(escaped_byte_iter(msg)),
        Value::List(elts) => return ValueIterator::List(elts.into_iter()),
        Value::Quotation(body) => return ValueIterator::Quotation(body.into_iter()),
    };
}

fn stack_copy<'a>(stack: &mut Vec<Value<'a>>,count: i64) {
    if count >= 0 {
        let offset = (count + 1) as usize;
        let val = if stack.len() >= offset {
            stack[stack.len() - offset].clone()
        }else {
            Value::default()
        };
        stack.push(val);
    } else {
        let offset = (1 - count) as usize;
        let val = stack.last().map(|v|v.clone()).unwrap_or_default();
        while stack.len() < offset {
            stack.insert(0,Value::default());
        }
        stack.insert(stack.len() - offset,val);
    }
}
fn stack_rot<'a>(stack: &mut Vec<Value<'a>>,count: i64) {
    if count == 0 {return;}
    if count > 0 {
        let offset = count as usize + 1;
        let src = if offset > stack.len() {
            Value::default()
        } else {
            stack.remove(stack.len() - offset)
        };
        stack.push(src)
    } else {
        let offset = -count as usize;
        let src = stack.pop().unwrap_or_default();
        while stack.len() < offset {
            stack.insert(0,Value::default());
        }
        stack.insert(stack.len() - offset,src);
    }
}
/*
if count is positive the top count elements are discarded
if count is negative the top -count elements are copied
*/
fn stack_drop_or_dup<'a>(stack: &mut Vec<Value<'a>>,count: i64) {
    if count >= 0 {
        for _ in 0..count {
            stack.pop();
        }
    } else {
        let offset = (-count) as usize;
        while stack.len() < offset {
            stack.insert(0,Value::default());
        }
        for i in stack.len() - offset..(stack.len()) {
            stack.push(stack[i].clone())
        }
    }
}
fn op_collect<'a>(stack: &mut Vec<Value<'a>>,count: i64) {
    let elt_count = count.abs() as usize;
    let mut elts = Vec::with_capacity(elt_count);
    if elt_count <= stack.len() {
        elts.extend_from_slice(&stack[(stack.len() - elt_count)..]);
    } else {
        elts.extend(std::iter::repeat(Value::default()).take(elt_count-stack.len()));
        elts.extend_from_slice(&stack);
    }
    stack.truncate(stack.len().saturating_sub(elt_count));
    if count < 0 {
        elts.reverse();
    }
    stack.push(Value::List(elts));
}

fn eval_buitlt_in<'a>(built_in: BuiltIn,stack: &mut Vec<Value<'a>>,globals: &mut HashMap<&'a [u8],Value<'a>>) {
    match built_in {
        BuiltIn::DUP => {
            stack_copy(stack,0);
        }
        BuiltIn::DUP2 => {
            stack_drop_or_dup(stack,-2);
        }
        BuiltIn::OVER => {
            stack_copy(stack,1);
        }
        BuiltIn::OVER2 => {
            stack_copy(stack,2);
        }
        BuiltIn::UNDER => {
            stack_copy(stack,-1);
        }
        BuiltIn::OVER_N => {
            let count = stack.pop().unwrap_or_default();
            match count {
                Value::Number(num) => {
                    stack_copy(stack,num.round() as i64);
                }
                Value::ByteString(_val) => {panic!("unimplemented")}
                Value::List(_elts) => {panic!("unimplemented")}
                Value::Quotation(_body) => {panic!("unimplemented")}
            }
        }
        BuiltIn::SWAP => {
            stack_rot(stack,1);
        }
        BuiltIn::SWAP2 => {
            stack_rot(stack,3);
            stack_rot(stack,3);
        }
        BuiltIn::ROT2 => {
            stack_rot(stack,2);
        }
        BuiltIn::ROT_2 => {
            stack_rot(stack,-2);
        }
        BuiltIn::ROT3 => {
            stack_rot(stack,3);
        }
        BuiltIn::ROT_3 => {
            stack_rot(stack,-3);
        }
        BuiltIn::ROT_N => {
            let count = stack.pop().unwrap_or_default();
            match count {
                Value::Number(num) => {
                    stack_rot(stack,num.round() as i64);
                }
                Value::ByteString(_val) => {panic!("unimplemented")}
                Value::List(_elts) => {panic!("unimplemented")}
                Value::Quotation(_body) => {panic!("unimplemented")}
            }
        }
        BuiltIn::DROP => {
            stack_drop_or_dup(stack,1);
        }
        BuiltIn::DROP2 => {
            stack_drop_or_dup(stack,2);
        }
        BuiltIn::DROP_N => {
            let count = stack.pop().unwrap_or_default();
            match count {
                Value::Number(num) => {
                    stack_drop_or_dup(stack,num.round() as i64);
                }
                Value::ByteString(_val) => {panic!("unimplemented")}
                Value::List(_elts) => {panic!("unimplemented")}
                Value::Quotation(_body) => {panic!("unimplemented")}
            }
        }
        BuiltIn::NEGATE => {
            unary_number_op(stack,|num|Value::Number(-num));
        }
        BuiltIn::ADD => {
            binary_number_op(stack,|left_num,right_num|Value::Number(left_num+right_num));
        }
        BuiltIn::SUBTRACT => {
            binary_number_op(stack,|left_num,right_num|Value::Number(left_num-right_num));
        }
        BuiltIn::MULTIPLY => {
            binary_number_op(stack,|left_num,right_num|Value::Number(left_num*right_num));
        }
        BuiltIn::DIVIDE => {
            binary_number_op(stack,|left_num,right_num|Value::Number(left_num/right_num));
        }
        BuiltIn::MODULO => {
            binary_number_op(stack,|left_num,right_num|Value::Number(left_num%right_num));
        }
        BuiltIn::LEN => {
            let val = stack.pop().unwrap_or_default();
            let length = match val {
                // TODO is this a useful value
                Value::Number(num) => (num.abs()+1.0).log2().ceil()+(if num<0. {1.} else {0.}),
                Value::ByteString(val) => val.len() as f64,
                Value::List(elts) => elts.len() as f64,
                Value::Quotation(body) => body.len() as f64
            };
            stack.push(Value::Number(length));
        }
        BuiltIn::COLLECT1 => {
            op_collect(stack,1);
        }
        BuiltIn::COLLECT2 => {
            op_collect(stack,2);
        }
        BuiltIn::COLLECT => {
            let size = stack.pop().unwrap_or_default();
            match size {
                Value::Number(num) => {
                    op_collect(stack,num.round_ties_even() as i64);
                }
                Value::ByteString(_val) => {panic!("collect count cannot be a list");}
                Value::List(_elts) => {panic!("collect count cannot be a list");}
                Value::Quotation(_body) => {
                    panic!("collect count cannot be a quotation");
                }
            };
        }
        BuiltIn::IF => {
            let else_body = stack.pop().unwrap_or_default();
            let if_body = stack.pop().unwrap_or_default();
            let condition = stack.pop().unwrap_or_default();
            if as_bool(condition) {
                eval_or_value(&if_body,stack,globals)
            } else {
                eval_or_value(&else_body,stack,globals)
            }
        }
        BuiltIn::REPEAT | BuiltIn::FOR | BuiltIn::MAP => {
            let body = stack.pop().unwrap_or_default();
            let counter = stack.pop().unwrap_or_default();
            let map_stack = Vec::new();
            if built_in == BuiltIn::MAP {
                stack.push(Value::List(map_stack))
            }
            let val_stack = if built_in == BuiltIn::MAP {
                match stack.last_mut().unwrap() {
                    Value::List(elts) => elts,
                    _ => panic!("last element of stack should be a list")
                }
            } else {stack};
            for elt in as_iter(counter) {
                if built_in != BuiltIn::REPEAT {
                    val_stack.push(elt);
                }
                eval_or_value(&body,val_stack,globals)
            };
        }
        BuiltIn::CALL => {
            let val = stack.pop().unwrap_or_default();
            eval_call(&val,stack,globals);
        }
        _ => panic!("unimplemented: {}",String::from_utf8_lossy(builtin_name(&built_in).unwrap()))
    }
}
fn eval_block<'a>(atoms: &'a [Atom<'a>],stack: &mut Vec<Value<'a>>,globals: &mut HashMap<&'a [u8],Value<'a>>){
    for atom in atoms {
        match atom {
            Atom::Number(num) => {
                stack.push(Value::Number(*num))
            }
            Atom::CharString(val) => {
                escaped_char_iter(val).for_each(|c|stack.push(Value::Number(c as i32 as f64)))
            }
            Atom::ByteString(val) => {
                stack.push(Value::ByteString(val))
            }
            Atom::Quotation(body) => {
                stack.push(Value::Quotation(body))
            }
            Atom::BuiltInWord(built_in) => {
                eval_buitlt_in(*built_in,stack,globals);
            }
            Atom::Assign(name) => {
                globals.insert(name,stack.pop().unwrap_or_default());
            }
            Atom::Get(name) => {
                stack.push(globals.get(name).map(|v|v.clone()).unwrap_or_default());
            }
        }
    }
}
fn run_program<'a>(atoms: &'a [Atom<'a>]) -> Vec<Value<'a>> {
    let mut stack = Vec::<Value<'a>>::new();
    let mut globals = HashMap::new();
    eval_block(atoms,&mut stack,&mut globals);
    return stack;
}
// TODO conversion  atoms <-> id-list <-> byte-stream (multiple encodings for optimizing golfability)
// optimizations
// * special cases for blocks or strings of length 1 and 2
// * compound tokens for common atom-sequences
// * allow skipping brackets at start/end
// * encoding as bytes or nibbles in UTF-8 like format (without the redundancy)
// 0000 ... 0111 , 1000 0000 ... 1011 1111 , 1100 0000 0000 ... 1101 1111 1111 , ...

// main
fn main() -> io::Result<()> {
    // Read the content of the input file
    let input = fs::read("in.txt")?;

    let tokens = tokenize(&input);
    let atoms = parse_program(&tokens);

    // Write the output to the output file
    let mut out_file = fs::File::create("tokens.txt")?;
    for token_string in tokens.iter().map(|token| token.to_string()) {
        writeln!(out_file, "{}", token_string)?;
    }
    out_file = fs::File::create("atoms.txt")?;
    dump_atoms(&mut out_file,&atoms,0)?;

    let result = run_program(&atoms);
    for val in result.iter().rev() {
        writeln!(stdout(),"{}",val.to_string())?;
    }
    Ok(())
}
