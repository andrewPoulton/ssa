use pyo3::pyclass;
use sentencepiece::*;
use std::collections::VecDeque;

pub fn load_model(path: &String) -> Option<SentencePieceProcessor> {
    if let Ok(sp_model) = SentencePieceProcessor::open(path) {
        Some(sp_model)
    } else {
        eprintln!("Warning: SentencePiece model not loaded (or not provided). Defaulting to byte-level tokenizer.");
        None
    }
}

/// Remove any non-alphanumeric or whitespace ascii character.
/// Works by filtering raw bytes (hence unsafe), using the fact that in a UTF-8 stream
/// any byte < 128 is guaranteed to map to its ascii equivalent.
/// Mutability in the input isn't actually used, it's only there to satisfy
/// type requirements when used in tokenization.
fn remove_ascii_punctuation(text: &mut str) -> String {
    unsafe {
        let text_bytes = text.as_bytes_mut();
        let out_string: Vec<u8> = text_bytes
            .iter()
            .filter_map(|b| {
                match *b {
                    // Tab, new line, carriage return, space
                    9 | 10 | 13 | 32 => Some(*b),
                    // 0-9
                    48..=57 => Some(*b),
                    // A-Z
                    65..=90 => Some(*b),
                    // a-z
                    97..=122 => Some(*b),
                    // any other UTF-8 byte
                    128..=255 => Some(*b),
                    _ => None,
                }
            })
            .collect();
        std::str::from_utf8_unchecked(out_string.as_ref()).to_string()
    }
}

/// Determine if a utf8 byte is indexable.
/// A byte is indexable if it appears at the start of a word or number (so the previous byte is not in [A-Za-z0-9]).
/// and is the first byte in a utf8 codepoint.
/// This heuristic is biased toward space-separated and punctuated text
fn is_indexable_utf8_byte(byte: u8, prev_byte: u8) -> bool {
    // determine if byte is first byte of codepoint
    if byte < 127 // ascii byte
        || byte >> 5 == 6  // byte = 110xxxxx
        || byte >> 4 == 14 // byte = 1110xxxx
        || byte >> 3 == 30
    // byte = 11110xxx
    {
        // if prev_byte is not ascii alnum, and not a continuation byte (10xxxxxx), mark as indexible.
        match prev_byte {
            0..=47 | 58..=64 | 91..=96 | 123..=127 => true,
            128..=255 => {
                // if byte = 1110xxxx or 11110xxx, it might be CJK etc (i.e. not space/punc delimited text)
                // or a multi-point character (eg an emoji)
                // so we assume that if the prev_byte is also a continuation byte, we should still index.
                // This is probably a bad heuristic and should be changed.

                // Note that we don't worry about whether prev_byte is a true continuation byte (i.e. 10xxxxxx),
                // since we assume valid utf8 and we'd be in the other branch if prev_byte was not a continuation byte.
                byte >> 4 == 14 || byte >> 3 == 30
            }
            _ => false,
        }
    } else {
        false
    }
}

#[derive(Debug)]
#[pyclass]
pub struct Tokenizer {
    pub sp_model: Option<SentencePieceProcessor>,
    pub width: u8,
}

impl Tokenizer {
    pub fn new(sp_model: Option<SentencePieceProcessor>) -> Self {
        let width = if let Some(ref sp) = sp_model {
            if sp.len() >= (1 << 16) {
                2
            } else {
                4
            }
        } else {
            1
        };
        Tokenizer { sp_model, width }
    }

    fn _encode_bytes(&self, text: &str) -> Result<Vec<PieceWithId>, SentencePieceError> {
        let mut byte_pieces: Vec<PieceWithId> = Vec::with_capacity(text.len());

        fn bytepiece(byte: u8, index: bool) -> PieceWithId {
            if index {
                PieceWithId {
                    piece: String::from("▁"),
                    id: byte as u32,
                    span: (0, 0),
                }
            } else {
                PieceWithId {
                    piece: String::new(),
                    id: byte as u32,
                    span: (0, 0),
                }
            }
        }
        let text = text.as_bytes();
        // Always index the first byte, text is always non-empty
        byte_pieces.push(bytepiece(text[0], true));
        for i in 1..text.len() {
            let is_indexible = is_indexable_utf8_byte(text[i], text[i - 1]);
            let next_piece = bytepiece(text[i], is_indexible);
            byte_pieces.push(next_piece);
        }
        Ok(byte_pieces)
    }

    fn _encode_sp(&self, text: &str) -> Result<Vec<PieceWithId>, SentencePieceError> {
        self.sp_model.as_ref().unwrap().encode(text)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<PieceWithId>, SentencePieceError> {
        if self.sp_model.is_some() {
            self._encode_sp(text)
        } else {
            self._encode_bytes(text)
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct IndexableTokens {
    #[pyo3(get)]
    pub tokens: Vec<u8>,
    #[pyo3(get)]
    pub no_index: Vec<u64>,
}

pub enum TokenDelim {
    BOS,
    EOS,
    BOTH,
    NEITHER,
}

impl TokenDelim {
    pub fn from(i: i32) -> Self {
        match i {
            0 => TokenDelim::BOS,
            1 => TokenDelim::EOS,
            _ => TokenDelim::NEITHER,
        }
    }
}

impl IndexableTokens {
    pub fn from_query(
        tokenizer: &Tokenizer,
        query: &mut str,
        delim: TokenDelim,
        lowercase: bool,
        remove_punc: bool,
    ) -> Self {
        if lowercase {
            query.make_ascii_lowercase();
        }

        let pieces = if remove_punc {
            let q = remove_ascii_punctuation(query);
            tokenizer.encode(&q).unwrap()
        } else {
            tokenizer.encode(&query).unwrap()
        };
        let no_index = pieces
            .iter()
            .enumerate()
            .fold(Vec::new(), |mut acc, (idx, p)| {
                // Sentencepiece identifies tokens at the start of words with an `U+2581` character.
                // We push all pieces without this character onto the no_index vec.
                if !p.piece.starts_with("▁") {
                    acc.push((idx as u64) * 2);
                }
                acc
            });
        // We don't need to write out the tokens if we're using byte tokenization
        let tokens = if tokenizer.sp_model.is_some() {
            let mut tokens_: VecDeque<u8> = pieces
                .iter()
                .map(|x| {
                    if tokenizer.width == 2 {
                        (x.id as u16).to_le_bytes().to_vec()
                    } else {
                        x.id.to_le_bytes().to_vec()
                    }
                })
                .flatten()
                .collect();
            match delim {
                TokenDelim::BOS => {
                    for t in (tokenizer.sp_model.as_ref().unwrap().bos_id().unwrap() as u16).to_be_bytes() {
                        tokens_.push_front(t)
                    }
                }
                TokenDelim::EOS => {
                    for t in (tokenizer.sp_model.as_ref().unwrap().eos_id().unwrap() as u16).to_le_bytes() {
                        tokens_.push_back(t)
                    }
                }
                _ => (),
            };
            tokens_.make_contiguous().to_vec()
        } else {
            Vec::new()
        };
        Self { tokens, no_index }
    }
}

pub fn tokenizer_encode(tokenizer: &String, query: &String) -> Result<Vec<u32>, ()> {
    let tokenizer = load_model(tokenizer).unwrap();
    let tokens = tokenizer.encode(query).unwrap();
    let view_tokes: Vec<u32> = tokens.iter().map(|x| x.id).collect();
    Ok(view_tokes)
}
