use std::fs;
use std::io::{self, Write,stdout};
use std::collections::HashMap;
use std::sync::OnceLock;
use std::ops::Range;
use std::str::Bytes;
use std::vec::IntoIter;
use std::slice;

// tokenizer
#[derive(Debug,PartialEq,Clone,Copy)]
enum TokenType{
    Word,
    String,
    OpenBracket,
    ClosingBracket,
    EOF
}
impl ToString for TokenType {
    fn to_string(&self) -> String {
        match self {
            TokenType::Word => "Word",
            TokenType::String => "String",
            TokenType::OpenBracket => "OpenBracket",
            TokenType::ClosingBracket => "ClosingBracket",
            TokenType::EOF => "EOF",
        }.to_string()
    }
}
const BRACKET_CHARS: &str = "{}";
fn is_special_char(op_char: char) -> bool {
    BRACKET_CHARS.contains(op_char)
}
fn is_open_bracket_char(op_char: char) -> bool {
    op_char=='{'
}
#[derive(Debug,Clone,Copy)]
struct Position{
    line :i32,
    line_pos: i32
}
#[derive(Debug,Clone,Copy)]
struct Token<'a>{
    token_type: TokenType,
    value: &'a str,
    pos: Position
}

impl ToString for Token<'_> {
    fn to_string(&self) -> String {
        format!("{:?}: \"{}\" at {}:{}", self.token_type, self.value, self.pos.line, self.pos.line_pos)
    }
}

fn tokenize<'a>(input: &'a str) -> Vec<Token<'a>> {
    // TODO support strings
    let mut start_index: usize=0;
    let mut line = 1;
    let mut line_pos = 0;
    let mut start_line = 1;
    let mut start_pos = 0;
    let mut output = Vec::new();
    for (i, c) in input.chars().enumerate() {
        if c == '\n' {
            line_pos = 0;
            line += 1;
        } else {
            line_pos += 1;
        }
        let is_operator = is_special_char(c);
        if c.is_whitespace() || is_operator {
            if start_index < i {
                let token_value = &input[start_index..i];
                output.push(Token{
                    token_type: TokenType::Word,
                    value: token_value,
                    pos: Position{line:start_line,line_pos: start_pos}
                });
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
        let first_char = token_value.chars().next().unwrap();
        let token_type = if is_open_bracket_char(first_char) {
            TokenType::OpenBracket
        } else if is_special_char(first_char) {
            TokenType::ClosingBracket
        } else {
            TokenType::Word
        };
        output.push(Token{
            token_type: token_type,
            value: "",
            pos: Position{line:start_line,line_pos: start_pos}
        });
    }
    output.push(Token{
        token_type: TokenType::EOF,
        value: token_value,
        pos: Position{line:line,line_pos: line_pos}
    });
    return output
}

// parser
#[derive(Debug,Clone,Copy)]
enum StackOp{
    Over(u32),
    Under(u32),
    Rotate(i32), // positive: rotate n-th element to top, negative: rotate top-element down to pos n
    Drop(u32), // discard up to n values
}

#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash)]
enum BuiltIn {
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
    SPLAT,

    IF,
    REPEAT,
    FOR,
    MAP,

    CALL,
}
const BUILTIN_WORDS: [(&str,BuiltIn); 32] = [
    // arithmetic operators
    ("-_", BuiltIn::NEGATE),
    ("negate", BuiltIn::NEGATE),
    ("+", BuiltIn::ADD),
    ("add", BuiltIn::ADD),
    ("-", BuiltIn::SUBTRACT),
    ("sub", BuiltIn::SUBTRACT),
    ("*", BuiltIn::MULTIPLY),
    ("mul", BuiltIn::MULTIPLY),
    ("/", BuiltIn::DIVIDE),
    ("div", BuiltIn::DIVIDE),
    ("%", BuiltIn::MODULO),
    ("mod", BuiltIn::MODULO),
    ("^", BuiltIn::POW),
    ("pow", BuiltIn::POW),
    // logic operators
    ("&", BuiltIn::AND),
    ("and", BuiltIn::AND),
    ("|", BuiltIn::OR),
    ("or", BuiltIn::OR),
    ("xor", BuiltIn::XOR),
    ("!", BuiltIn::NOT),
    ("not", BuiltIn::NOT),
    // lists
    ("len", BuiltIn::LEN),
    ("collect", BuiltIn::COLLECT),
    ("...", BuiltIn::SPLAT),
    ("splat", BuiltIn::SPLAT),
    // control flow
    ("?", BuiltIn::IF),
    ("if", BuiltIn::IF),
    ("repeat", BuiltIn::REPEAT),
    ("for", BuiltIn::FOR),
    ("map", BuiltIn::MAP),
    // functions
    ("()", BuiltIn::CALL),
    ("call", BuiltIn::CALL),
];
static BUILTIN_INFO: OnceLock<HashMap<&str,BuiltIn>> = OnceLock::new();
static BUILTIN_NAMES: OnceLock<HashMap<BuiltIn,&str>> = OnceLock::new();
fn builtin_id(word: &str) -> Option<BuiltIn> {
    BUILTIN_INFO.get_or_init(|| {
        let mut map = HashMap::new();
        for (key, value) in BUILTIN_WORDS {
            map.insert(key, value);
        }
        map
    }).get(word).map(|v|*v)
}
fn builtin_name(built_in: &BuiltIn) -> Option<&'static str> {
    BUILTIN_NAMES.get_or_init(|| {
        let mut map = HashMap::new();
        for (name, id) in BUILTIN_WORDS {
            map.insert(id, name);
        }
        map
    }).get(built_in).map(|v|*v)
}

#[derive(Debug,Clone)]
enum Atom<'a> {
    BuiltInWord(BuiltIn),
    Number(f64), // TODO? use arbitary pression library for numbers (? rug crate )
    StackOp(StackOp),
    Assign(&'a str),
    Get(&'a str),
    String(&'a str),
    Quotation(Vec<Atom<'a>>),
}
fn dump_atoms<'a>(out_file: &mut fs::File, atoms: &'a [Atom<'a>],indent: usize)-> io::Result<()> {
    for atom in atoms {
        match atom {
            Atom::BuiltInWord(built_in) => {
                writeln!(out_file,"{}BuiltIn({})","   ".repeat(indent),builtin_name(built_in).unwrap())?;
            }
            Atom::Number(value) => {
                writeln!(out_file,"{}Number({})","   ".repeat(indent),value)?;
            }
            Atom::StackOp(stack_op) => {
                match stack_op {
                  StackOp::Over(index) => {writeln!(out_file,"{}Over({})","   ".repeat(indent),index)?;}
                  StackOp::Under(index) => {writeln!(out_file,"{}Under({})","   ".repeat(indent),index)?;}
                  StackOp::Rotate(index) => {writeln!(out_file,"{}Rotate({})","   ".repeat(indent),index)?;}
                  StackOp::Drop(index) => {writeln!(out_file,"{}Drop({})","   ".repeat(indent),index)?;}
                }
            }
            Atom::String(value) => { // TODO? escaping
                writeln!(out_file,"{}String({})","   ".repeat(indent),value.to_string())?;
            }
            Atom::Quotation(elts) => {
                writeln!(out_file,"{}{{","   ".repeat(indent))?;
                dump_atoms(out_file,&elts,indent+1)?;
                writeln!(out_file,"{}}}","   ".repeat(indent))?;
            }
            Atom::Assign(name) => {
                writeln!(out_file,"{}Assign({})","   ".repeat(indent),name.to_string())?;
            }
            Atom::Get(name) => {
                writeln!(out_file,"{}Get({})","   ".repeat(indent),name.to_string())?;
            }
        }
    }
    Ok(())
}

fn try_parse_atom<'a>(mut tokens: &'a [Token<'a>]) -> Result<(Atom<'a>,usize),&'a Token<'a>> {   
    match tokens[0].token_type {
        TokenType::Word => {
            let word = tokens[0].value;
            if word.len() > 0 && word.chars().next().unwrap().is_digit(10) ||
                word.len() > 1 && word.chars().next().unwrap() == '-' &&
                    word[1..].chars().next().unwrap().is_digit(10) {
                // TODO support non-integer numbers
                if let Ok(value) = word.parse::<f64>() {
                    return Ok((Atom::Number(value),1))
                } else {
                    return Err(&tokens[0])
                }
            } else if word == "dup" {
                return Ok((Atom::StackOp(StackOp::Over(0)),1))
            } else if word == "swap" {
                return Ok((Atom::StackOp(StackOp::Rotate(1)),1))
            } else if word == "drop" || word.starts_with("drop") &&
                    word[4..].chars().next().unwrap().is_digit(10) {
                let count_or_err = if word.len() == 4 {Result::Ok(1)} else {(&word[4..]).parse::<u32>()};
                if let Ok(count) = count_or_err {
                    return Ok((Atom::StackOp(StackOp::Drop(count)),1))
                } else {
                    return Err(&tokens[0])
                }
            } else if word == "over" || word.starts_with("over") &&
                    word[4..].chars().next().unwrap().is_digit(10) {
                let offset_or_err = if word.len() == 4 {Result::Ok(1)} else {(&word[4..]).parse::<u32>()};
                if let Ok(offset) = offset_or_err {
                    return Ok((Atom::StackOp(StackOp::Over(offset)),1))
                } else {
                    return Err(&tokens[0])
                }
            } else if word == "under" || word.starts_with("under") &&
                    word[5..].chars().next().unwrap().is_digit(10) {
                let offset_or_err = if word.len() == 5 {Result::Ok(1)} else {(&word[5..]).parse::<u32>()};
                if let Ok(offset) = offset_or_err {
                    return Ok((Atom::StackOp(StackOp::Under(offset)),1))
                } else {
                    return Err(&tokens[0])
                }
            } else if word == "rot" || (word.starts_with("rot") &&
                    word[3..].chars().next().unwrap().is_digit(10)) || (word.starts_with("rot-") &&
                    word[4..].chars().next().unwrap().is_digit(10)) {
                let count_or_err = if word.len() == 3 {Result::Ok(2)} else {(&word[3..]).parse::<i32>()};
                if let Ok(count) = count_or_err {
                    return Ok((Atom::StackOp(StackOp::Rotate(count)),1))
                } else {
                    return Err(&tokens[0])
                }
            } else if word.len() > 0 && word.ends_with("=") {
                return Ok((Atom::Assign(&word[..(word.len()-1)]),1))
            } else if let Some(built_in) = builtin_id(word) {
                return Ok((Atom::BuiltInWord(built_in),1))
            }
            return Ok((Atom::Get(word),1))
        },
        TokenType::String => return Ok((Atom::String(tokens[0].value),1)),
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
    String(&'a str),
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
            Value::String(val) => {
                return format!("String({})",val);
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

fn binary_number_op<'a>(left: &Value<'a>,right: &Value<'a>,f: fn(f64,f64)->Value<'a>) -> Value<'a> {
    match left {
        Value::Number(left_num) => {
            match right {
                Value::Number(right_num) => {return f(*left_num,*right_num)}
                Value::String(_val) => {panic!("unsuported operands for binary operation");}
                Value::List(_elts) => {panic!("unsuported operands for binary operation");}
                Value::Quotation(_body) => {panic!("unsuported operands for binary operation");}
            }
        }
        Value::String(_val) => {panic!("unsuported operands for binary operation");}
        Value::List(_elts) => {panic!("unsuported operands for binary operation");}
        Value::Quotation(_body) => {panic!("unsuported operands for binary operation");}
    }
}
            
fn eval_call<'a>(val: &Value<'a>,stack: &mut Vec<Value<'a>>,globals: &mut HashMap<&'a str,Value<'a>>) {
    match val {
        Value::Number(_num) => {panic!("cannot call number");} // ? create list with given number of elements
        Value::String(_val) => {panic!("cannot call string");} // TODO? eval string as code
        Value::List(_elts) => {panic!("cannot call list");} // TODO? call for each element (with current stack)
        Value::Quotation(body) => {
            eval_block(body,stack,globals)
        }
    }
}
fn eval_or_value<'a>(val: &Value<'a>,stack: &mut Vec<Value<'a>>,globals: &mut HashMap<&'a str,Value<'a>>) {
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
        Value::String(val) => val.len() != 0,
        Value::List(elts) => elts.len() != 0,
        Value::Quotation(body) => body.len() != 0
    }
}

enum ValueIterator<'a> {
    Number(Range<i64>),
    String(Bytes<'a>),
    List(IntoIter<Value<'a>>),
    Quotation(slice::Iter<'a, Atom<'a>>),
}
impl<'a> Iterator for ValueIterator<'a> {
   type Item = Value<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ValueIterator::Number(range) => range.next().map(|i| Value::Number(i as f64)),
            ValueIterator::String(iter) => iter.next().map(|b|Value::Number(b as f64)),
            ValueIterator::List(iter) => iter.next(),
            ValueIterator::Quotation(iter) => iter.next().map(|x|Value::Quotation(slice::from_ref(x))),
        }
    }
}
fn as_iter(val : Value) -> ValueIterator {
    match val {
        // TODO how to handle negatives
        Value::Number(num) => return ValueIterator::Number(0..(num as i64)), //TODO? rounding
        Value::String(msg) => return ValueIterator::String(msg.bytes()),
        Value::List(elts) => return ValueIterator::List(elts.into_iter()),
        Value::Quotation(body) => return ValueIterator::Quotation(body.into_iter()),
    };
}

fn eval_buitlt_in<'a>(built_in: BuiltIn,stack: &mut Vec<Value<'a>>,globals: &mut HashMap<&'a str,Value<'a>>) {
    match built_in {
        BuiltIn::ADD => {
            let right = stack.pop().unwrap_or_default();
            let left = stack.pop().unwrap_or_default();
            stack.push(binary_number_op(&left,&right,|left_num,right_num|Value::Number(left_num+right_num)));
        }
        BuiltIn::MULTIPLY => {
            let right = stack.pop().unwrap_or_default();
            let left = stack.pop().unwrap_or_default();
            stack.push(binary_number_op(&left,&right,|left_num,right_num|Value::Number(left_num*right_num)));
        }
        BuiltIn::DIVIDE => {
            let right = stack.pop().unwrap_or_default();
            let left = stack.pop().unwrap_or_default();
            stack.push(binary_number_op(&left,&right,|left_num,right_num|Value::Number(left_num/right_num)));
        }
        BuiltIn::LEN => {
            let val = stack.pop().unwrap_or_default();
            let length = match val {
                // TODO is this a useful value
                Value::Number(num) => (num.abs()+1.0).log2().ceil()+(if num<0. {1.} else {0.}),
                Value::String(val) => val.len() as f64,
                Value::List(elts) => elts.len() as f64,
                Value::Quotation(body) => body.len() as f64
            };
            stack.push(Value::Number(length));
        }
        BuiltIn::COLLECT => {
            let size = stack.pop().unwrap_or_default();
            match size {
                Value::Number(num) => {
                    let count = num as i64;//TODO? rounding
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
                Value::String(val) => {panic!("collect count cannot be a list");}
                Value::List(elts) => {panic!("collect count cannot be a list");}
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
        _ => panic!("unimplemented: {}",builtin_name(&built_in).unwrap())
    }
}
fn eval_block<'a>(atoms: &'a [Atom<'a>],stack: &mut Vec<Value<'a>>,globals: &mut HashMap<&'a str,Value<'a>>){
    for atom in atoms {
        match atom {
            Atom::Number(num) => {
                stack.push(Value::Number(*num))
            }
            Atom::String(val) => {
                stack.push(Value::String(val))
            }
            Atom::Quotation(body) => {
                stack.push(Value::Quotation(body))
            }
            Atom::StackOp(stack_op) => {
                match stack_op {
                    StackOp::Drop(count) => {
                        for _ in 0..*count {
                            stack.pop();
                        }
                    }
                    StackOp::Over(count) => {
                            let offset = *count as usize + 1;
                            let src = if offset > stack.len() {
                                Value::default()
                            } else {
                                stack.get(stack.len() - offset).unwrap().clone()
                            };
                            stack.push(src)
                    }
                    StackOp::Under(count) => {
                        let offset = *count as usize + 1;
                        let src = stack.last().map(|v|v.clone()).unwrap_or_default();
                        while stack.len() < offset {
                            stack.insert(0,Value::default());
                        }
                        stack.insert(stack.len() - offset,src);
                    }
                    StackOp::Rotate(count) => {
                        if *count == 0 {return;}
                        if *count > 0 {
                            let offset = *count as usize + 1;
                            let src = if offset > stack.len() {
                                Value::default()
                            } else {
                                stack.remove(stack.len() - offset)
                            };
                            stack.push(src)
                        } else {
                            let offset = -*count as usize;
                            let src = stack.pop().unwrap_or_default();
                            while stack.len() < offset {
                                stack.insert(0,Value::default());
                            }
                            stack.insert(stack.len() - offset,src);
                        }
                    }
                }
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
// TODO create interperter

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
    let input = fs::read_to_string("in.txt")?;

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
