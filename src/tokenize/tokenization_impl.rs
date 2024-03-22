use crate::lib_utils::{get_part_offsets, to_bytes};
use crate::tokenize::tokenizer;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde_json::Error;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

pub fn multithread_tokenize_jsonl(
    text_path: String,
    out_path: String,
    text_key: String,
    model: String,
    total_jobs: usize,
    delim: i32,
    lowercase: bool,
    remove_punc: bool,
    show_progress: usize,
) -> Result<(), Error> {
    let offsets = find_tokenizer_boundaries(&text_path, total_jobs);
    let styles = [
        ("Rough bar:", "█  ", "red"),
        ("Fine bar: ", "█▉▊▋▌▍▎▏  ", "yellow"),
        ("Vertical: ", "█▇▆▅▄▃▂▁  ", "green"),
        ("Fade in:  ", "█▓▒░  ", "blue"),
        ("Blocky:   ", "█▛▌▖  ", "magenta"),
    ];
    let m = MultiProgress::new();

    let result: Vec<_> = (0..total_jobs)
        .into_iter()
        .map(|i| {
            let (start, end) = offsets[i];
            let path_ = text_path.clone();
            let text_key_ = text_key.clone();
            let model_ = model.clone();
            let outpath_ = out_path.clone();
            // for clarity, show a max of 32 progress bars
            // let show_pb_modulo = 0;//total_jobs / total_jobs;
            let pb: Option<ProgressBar> = if i < show_progress {
                Some(m.add(ProgressBar::new((end - start) as u64)))
            } else {
                None
            };

            let style = styles[i % 5];
            if pb.is_some() {
                pb.as_ref().unwrap().set_prefix(format!("Thread {i:0>3}: {start}-{end}"));
                pb.as_ref().unwrap().set_style(
                    ProgressStyle::with_template(&format!(
                        // "{{prefix:.bold}} [{{elapsed_precise}}]▕{{bar:.{}}}▏{{pos:>9}}/{{len:9}}",
                        "{{prefix:.bold}} [{{elapsed_precise}} ({{per_sec}})]▕{{bar:.{}}}▏{{percent}}% {{pos:>9}}/{{len:>9}} | {{msg}}",

                        style.2
                    ))
                    .unwrap()
                    .progress_chars(style.1),
                );
            }
            let one_result: std::thread::JoinHandle<String> = std::thread::spawn(move || {

                let tokenize_output = tokenize_part(path_, model_, text_key_, start, end, delim, lowercase, remove_punc, pb);

                match tokenize_output {
                    Ok((tokens, sample_index)) => {

                            let token_outpath = format!("{}.tokens.{:0>5}", &outpath_, i);
                            let noindex_outpath = format!("{}.noindex.{:0>5}", &outpath_, i);
                            let sample_index_outpath = format!("{}.sindex.{:0>5}", &outpath_, i);
                            let mut token_buffer =
                                std::io::BufWriter::new(fs::File::create(token_outpath).unwrap());
                            let mut noindex_buffer =
                                std::io::BufWriter::new(fs::File::create(noindex_outpath).unwrap());
                            let mut sample_index_buffer =
                                std::io::BufWriter::new(fs::File::create(sample_index_outpath).unwrap());
                            token_buffer
                                .write_all(&tokens.tokens[..])
                                .expect("Tokens didn't write");
                            sample_index_buffer
                                .write_all(&to_bytes(&sample_index[..], 8)[..])
                                .expect("Sample index didn't write");
                            if let Some(max_elt) = tokens.no_index.last() {
                                let width = ((*max_elt as f64).log2() / 8.0).ceil() as usize;
                                noindex_buffer
                                    .write_all(&to_bytes(&tokens.no_index[..], width)[..])
                                    .expect("Noindex didn't write");
                            }

                        let complete_msg = format!("Thread {i} (slice {start} - {end}) completed successfully\n");
                        complete_msg
                    },
                    Err(e) => {
                        e.to_owned()
                    }
                }
            });
            one_result
        })
        .collect();

    let mut full_msg = String::new();
    for (i, r) in result.into_iter().enumerate() {
        match r.join() {
            Ok(msg) => {
                full_msg.push_str(&msg);
            }
            _ => (),
        }
    }

    print!("{full_msg}");

    Ok(())
}

fn tokenize_part(
    file: String,
    model: String,
    text_key: String,
    start: usize,
    end: usize,
    delim: i32,
    lowercase: bool,
    remove_punc: bool,
    pb: Option<ProgressBar>,
) -> Result<(tokenizer::IndexableTokens, Vec<u64>), String> {
    let file = fs::File::open(file).unwrap();
    let mut reader = BufReader::new(file);
    let _ = reader.seek(std::io::SeekFrom::Start(start as u64));
    let mut tokens = Vec::new();
    let mut bytes_read = start;
    let mut total_tokens = 0u64;
    let sp_model = tokenizer::load_model(&model);
    let tokenizer = tokenizer::Tokenizer::new(sp_model);
    let mut sample_index = Vec::<u64>::new();
    while bytes_read < end {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line).unwrap();
        bytes_read += bytes;
        if let Ok(deser) = serde_json::from_str::<serde_json::Value>(&line) {
            if let serde_json::Value::String(text) = &deser[&text_key] {
                let delim = tokenizer::TokenDelim::from(delim);
                let line_tokens = tokenizer::IndexableTokens::from_query(
                    &tokenizer,
                    &mut text.to_string(),
                    delim,
                    lowercase,
                    remove_punc,
                );
                total_tokens += (line_tokens.tokens.len() / 2) as u64;
                tokens.push(line_tokens);
            }
            if pb.is_some() {
                pb.as_ref().unwrap().inc(bytes as u64);
            }
        }
        sample_index.push(total_tokens);
        sample_index.push(bytes_read as u64);
    }
    if pb.is_some() {
        pb.as_ref().unwrap().finish();
    }
    let (result, _) = tokens.iter().fold(
        (
            tokenizer::IndexableTokens {
                tokens: Vec::new(),
                no_index: Vec::new(),
            },
            0,
        ),
        |mut acc, it| {
            acc.0.tokens.extend(it.tokens.iter());
            let _: Vec<_> = it
                .no_index
                .iter()
                .map(|x| acc.0.no_index.push(acc.1 + x))
                .collect();
            acc.1 += it.tokens.len() as u64;
            acc
        },
    );
    Ok((result, sample_index))
}

pub fn find_tokenizer_boundaries(text_path: &String, total_jobs: usize) -> Vec<(usize, usize)> {
    // we want the tokenizer
    let total_len = std::fs::metadata(text_path.to_string()).unwrap().len();
    let mut offsets = get_part_offsets(total_len as usize, 0, total_jobs);
    let mut file = fs::File::open(text_path.to_string()).unwrap();
    let mut reader = BufReader::new(file);
    // println!("Loading part of file from byte {} to {}", start, end);
    let mut line = String::new();
    // let l = jsonl::read(file);
    for i in 0..offsets.len() - 1 {
        let mut byte_offset = offsets[i].1; // end byte offset for this section
        let _ = reader.seek(std::io::SeekFrom::Start(byte_offset as u64));

        let mut line = Vec::new();
        byte_offset += loop {
            if let Ok(bytes) = reader.read_until(10u8, &mut line) {
                break bytes;
            } else {
                byte_offset += line.len();
                line.clear();
            }
        };

        byte_offset += reader
            .read_until(10u8, &mut line)
            .unwrap_or(total_len as usize);
        offsets[i].1 = byte_offset;
        offsets[i + 1].0 = byte_offset;
    }

    offsets
}

pub struct SampleIndex {
    index: BufReader<File>,
    cache: [u8; 32],
    size: u64,
}
#[derive(Debug)]
pub enum IndexSearchError {
    OutOfBounds,
    MaxItsExceeded,
}

impl SampleIndex {
    pub fn new(path: String) -> Self {
        let size = std::fs::metadata(&path).unwrap().len() / 16;
        let index = SampleIndex {
            index: std::io::BufReader::with_capacity(16, fs::File::open(&path).unwrap()),
            cache: [0u8; 32],
            size: size,
        };
        // table.seek (offset ).expect ("Seek failed!");
        index
    }

    fn seek(&mut self, pos: u64) -> std::io::Result<u64> {
        self.index.seek(std::io::SeekFrom::Start(pos * 16))
    }

    fn parse_cache(&self) -> [u64; 4] {
        self.cache
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<u64>>()
            .try_into()
            .unwrap()
    }

    pub fn search(&mut self, val: u64) -> Result<(u64, u64), IndexSearchError> {
        // Short circuit
        let _ = self.seek(0);
        let _ = self.index.read_exact(&mut self.cache);
        let first_val = self.parse_cache();
        if val < first_val[0] {
            return Ok((0, first_val[1]));
        }

        let mut lo = 1u64;
        let mut hi = self.size - 1;
        let _ = self.seek(hi - 1);
        let _ = self.index.read_exact(&mut self.cache);
        let first_val = self.parse_cache();
        if val >= first_val[2] {
            // println!("cache: {:?}, val: {}", first_val, val);
            return Err(IndexSearchError::OutOfBounds);
        }
        // let mut targ = self.size + 1;
        let max_its = (hi as f64).log2().ceil() as i32 + 10;
        let mut its = 1;
        loop {
            let mid = (hi + lo) / 2;
            let _ = self.seek(mid - 1);
            let _ = self.index.read_exact(&mut self.cache);
            let [prev_val, prev_offset, this_val, this_offset] = self.parse_cache();
            // println!("it: {}, lo: {}, mid: {}, hi: {}, cache: [{},{},{},{}], val: {}", its, lo, mid, hi, prev_val, prev_offset, this_val, this_offset, val);
            if this_val < val {
                lo = mid;
            } else if prev_val > val {
                hi = mid;
            } else {
                // prev_val <= val  <= this_val
                break Ok((mid - 1, prev_offset));
            }
            its += 1;
            if its > max_its {
                break Err(IndexSearchError::MaxItsExceeded);
            }
        }
    }
}
