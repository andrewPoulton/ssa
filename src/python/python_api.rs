use crate::off_disk_utils::read_file_to_buffer;
use crate::table::{
    build_in_memory_array_impl, find_duplicates_impl, find_off_disk_file_impl, find_off_disk_impl,
    in_memory_merge_impl,
};
use crate::tokenize::tokenizer::load_model;
use crate::tokenize::{find_tokenizer_boundaries, multithread_tokenize_jsonl, SampleIndex};
use crate::{from_bytes, get_part_offsets, index_to_hashset};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyString};
use serde::Serialize;
use std::collections::HashSet;
use std::fs;
use std::io::prelude::*;
use std::io::BufReader;
use std::thread;

#[pyfunction]
fn build_in_memory_array(
    text: &PyBytes,
    num_parts: usize,
    num_threads: usize,
    hacksize: usize,
) -> PyResult<(Vec<u8>, usize)> {
    // const hacksize: usize = 100_000;
    // Process:
    //  get starts/ends
    //  split text accordingly into vev<vec<u8>>
    //  iterate over job_params in num_thread chunks
    //  collect resultant tables
    //  merge
    let text = text.as_bytes();
    let exclude = HashSet::new();
    Ok(build_in_memory_array_impl(
        text,
        num_parts,
        num_threads,
        hacksize,
        &exclude,
    ))
}

#[pyfunction]
fn build_index_filtered_array_from_disk(
    token_path: &PyString,
    index_path: &PyString,
    // out_path: &PyString,
    // num_files: i32,
    num_parts: usize,
    num_threads: usize,
    hacksize: usize,
) -> PyResult<()> {
    let token_len = std::fs::metadata(token_path.to_string()).unwrap().len();
    let index_len = std::fs::metadata(index_path.to_string()).unwrap().len();
    let mut token_buffer: Vec<u8> = vec![0; token_len as usize];
    let mut index_buffer: Vec<u8> = vec![0; index_len as usize];
    let tokens = fs::File::open(token_path.to_string()).unwrap();
    let index = fs::File::open(index_path.to_string()).unwrap();
    let mut tokens_reader = BufReader::new(tokens);
    let mut index_reader = BufReader::new(index);
    tokens_reader.read_exact(&mut token_buffer[..])?;
    index_reader.read_exact(&mut index_buffer[..])?;
    let exclude = index_to_hashset(index_buffer);
    let (mut array, ratio) = build_in_memory_array_impl(
        &token_buffer[..],
        num_parts,
        num_threads,
        hacksize,
        &exclude,
    );
    array.push(ratio as u8);
    let outpath = format!("{}.st", token_path.to_string());
    let mut array_buffer = std::io::BufWriter::new(fs::File::create(outpath).unwrap());

    array_buffer
        .write_all(&array[..])
        .unwrap_or_else(|_| println!("index write failed"));
    println!(
        "Partial array for {} successfully built",
        token_path.to_string()
    );
    Ok(())
}

// Merging functionality
/* COMMENT IS FROM ORIGINAL AUTHORS
 * Merge together M different suffix arrays (probably created with make-part).
 * That is, given strings S_i and suffix arrays A_i compute the suffix array
 * A* = make-suffix-array(concat S_i)
 * In order to do this we just implement mergesort's Merge operation on each
 * of the arrays A_i to construct a sorted array A*.
 *
 * This algorithm is *NOT A LINEAR TIME ALGORITHM* in the worst case. If you run
 * it on a dataset consisting entirely of the character A it will be quadratic.
 * Fortunately for us, language model datasets typically don't just repeat the same
 * character a hundred million times in a row. So in practice, it's linear time.
 *
 * There are thre complications here.
 *
 * As with selfsimilar_parallel, we can't fit all A_i into memory at once, and
 * we want to make things fast and so parallelize our execution. So we do the
 * same tricks as before to make things work.
 *
 * However we have one more problem. In order to know how to merge the final
 * few bytes of array S_0 into their correct, we need to know what bytes come next.
 * So in practice we make sure that S_{i}[-hacksize:] === S_{i+1}[:hacksize].
 * As long as hacksize is longer than the longest potential match, everything
 * will work out correctly. (I did call it hacksize after all.....)
 * In practice this works. It may not for your use case if there are long duplicates.
 */
#[pyfunction]
fn in_memory_merge(
    texts: Vec<Vec<u8>>,
    tables: Vec<Vec<u8>>,
    ratios: Vec<usize>,
    num_threads: i64,
    hacksize: usize,
) -> PyResult<(Vec<u8>, usize)> {
    Ok(in_memory_merge_impl(
        texts,
        tables,
        ratios,
        num_threads,
        hacksize,
    ))
}

#[pyfunction]
fn merge_from_disk(
    texts: &PyList,
    tables: &PyList,
    // ratios: Vec<usize>,
    num_threads: i64,
) -> PyResult<()> {
    let mut outpath = tables[0].to_string(); // *.tokens.00000.st
    outpath.replace_range(outpath.len() - 8.., "st"); // *.tokens.st
    let texts: Vec<Vec<u8>> = texts
        .iter()
        .map(|x| {
            let path = x.to_string();
            read_file_to_buffer(&path)
        })
        .collect();
    let mut tables: Vec<Vec<u8>> = tables
        .iter()
        .map(|x| {
            let path = x.to_string();
            read_file_to_buffer(&path)
        })
        .collect();
    let ratios: Vec<usize> = tables
        .iter_mut()
        .map(|t| t.pop().unwrap() as usize)
        .collect();
    let hacksize = 0; // We're merging across document boundaries, so we don't need to check suffixes that cross these boundaries
    let (mut big_array, big_ratio) =
        in_memory_merge_impl(texts, tables, ratios, num_threads, hacksize);
    let mut array_buffer = std::io::BufWriter::new(fs::File::create(outpath).unwrap());
    big_array.push(big_ratio as u8);
    array_buffer
        .write_all(&big_array[..])
        .unwrap_or_else(|_| println!("array write failed"));
    Ok(())
}

#[pyfunction]
fn get_idx(idx: &PyBytes, width: usize, loc: u64) -> PyResult<usize> {
    Ok(from_bytes(idx.as_bytes().to_vec(), width).partition_point(|x| *x < loc))
}

#[pyfunction]
fn find_duplicates(
    text1: &PyBytes,
    table1: &PyBytes,
    text2: &PyBytes,
    table2: &PyBytes,
    ratio1: usize,
    ratio2: usize,
    length_threshold: usize,
    num_threads: i64,
    find_max: bool,
    skipgram_budget: usize,
    eos: u8,
) -> PyResult<Vec<Vec<u8>>> {
    let text_len1 = text1.len().unwrap_or(0);
    let table_len1 = table1.len().unwrap_or(0);
    let text_len2 = text2.len().unwrap_or(0);
    let table_len2 = table2.len().unwrap_or(0);
    if (text_len1 == 0) || (table_len1 == 0) || (text_len2 == 0) || (table_len2 == 0) {
        // quick exit on errors
        return Ok(vec![vec![0], vec![0], vec![0], vec![0], vec![0], vec![0]]);
    }
    let res = find_duplicates_impl(
        text1.as_bytes(),
        table1.as_bytes(),
        text2.as_bytes(),
        table2.as_bytes(),
        ratio1,
        ratio2,
        length_threshold,
        num_threads,
        find_max,
        skipgram_budget,
        eos,
    );
    Ok(res)
}

#[pyfunction]
fn collect(
    text: &PyBytes,
    table: &PyBytes,
    duplicate_locs: &PyBytes,
    length_threshold: u64,
) -> PyResult<Vec<(u64, u64)>> {
    let size_text = text.len().unwrap();
    let size_table = table.len().unwrap();

    assert!(size_table % size_text == 0);
    let size_width = size_table / size_text;

    let mut output: Vec<u64> = from_bytes(duplicate_locs.as_bytes().to_vec(), size_width);
    output.sort_unstable();
    let mut ranges: Vec<(u64, u64)> = Vec::with_capacity(1000);
    let mut prev_start = output[0];
    let mut prev_end = output[0] + length_threshold;
    for ptr in &output[1..] {
        if ptr < &prev_end {
            prev_end = ptr + length_threshold;
        } else {
            ranges.push((prev_start, prev_end));
            prev_start = *ptr;
            prev_end = ptr + length_threshold;
        };
    }
    Ok(ranges)
}
#[derive(Debug, Serialize, Hash)]
#[pyclass]
pub struct LocationIndex {
    #[pyo3(get)]
    pub indices: Vec<u64>,
    #[pyo3(get)]
    pub offsets: Vec<u64>,
    #[pyo3(get)]
    pub locations: Vec<u64>,
}

impl LocationIndex {
    pub fn new(capacity: usize) -> Self {
        LocationIndex {
            indices: Vec::with_capacity(capacity),
            offsets: Vec::with_capacity(capacity),
            locations: Vec::with_capacity(capacity),
        }
    }
    pub fn dumps(&self) -> PyResult<String> {
        Ok(serde_json::to_string(self).unwrap())
    }
}

#[pyfunction]
pub fn find(
    query: &PyString,
    tokens: &PyString,
    tokenizer: &PyString,
    debug: bool,
) -> PyResult<LocationIndex> {
    let query = query.to_string();
    let tokens: String = tokens.to_string();
    let tokenizer = tokenizer.to_string();
    let print_query = query.clone();
    let print_filename: String = tokens.clone();
    let mut sindex = tokens.clone();
    let r = sindex.len() - "tokens".len();
    sindex.replace_range(r.., "sindex");
    let (occurances, locs) = find_off_disk_impl(&query, &tokens, &tokenizer);
    println!("Found {occurances} occurances of `{print_query}` in {print_filename}");
    let mut sample_index = SampleIndex::new(sindex);
    let mut location_index = LocationIndex::new(locs.len());
    let _: Vec<_> = locs
        .iter()
        .map(|loc| {
            let (idx, offset) = sample_index.search(*loc / 2).unwrap();
            location_index.indices.push(idx);
            location_index.offsets.push(offset);
            location_index.locations.push(*loc);
        })
        .collect();
    if debug {
        let tokenizer = load_model(&tokenizer).unwrap();
        let pieces = tokenizer.encode(&query).unwrap();
        let tokens: Vec<u32> = pieces.iter().map(|p| p.id).collect();
        println!("Debug info:\n\tPieces: {pieces:?}\n\tTokens: {tokens:?}")
    }
    Ok(location_index)
}

#[pyfunction]
pub fn find_multi(query: &PyString, tokens: &PyList, tokenizer: &PyString, debug: bool) -> () {
    let mut print_query = query.to_string();
    if print_query.len() > 60 {
        print_query.replace_range(57.., "...");
    }
    let total_occurances: u64 = tokens
        .iter()
        .map(|tokens_| {
            let query_ = query.to_string();
            let tokenizer_ = tokenizer.to_string();
            let tokens_ = tokens_.to_string();
            let print_filename = tokens_.clone();
            let one_result =
                thread::spawn(move || find_off_disk_impl(&query_, &tokens_, &tokenizer_));

            if let Ok((occurances, _locs)) = one_result.join() {
                if debug {
                    println!(
                        "\tFound {occurances} occurances of `{print_query}` in {print_filename}"
                    );
                }
                occurances
            } else {
                0
            }
        })
        .collect::<Vec<u64>>()
        .iter()
        .sum();
    println!("Found {total_occurances} total occurances of `{print_query}`");
}

#[pyfunction]
fn find_off_disk(
    query_file: &PyString,
    tokens: &PyString,
    cache_dir: &PyString,
    length_threshold: usize,
    tokenizer_jobs: i64,
    linear_search: bool,
) -> () {
    find_off_disk_file_impl(
        &query_file.to_string(),
        &tokens.to_string(),
        &cache_dir.to_string(),
        length_threshold,
        tokenizer_jobs,
        linear_search,
    );
}

#[pyfunction]
pub fn tokenize_jsonl(
    text_path: &PyString,
    out_path: &PyString,
    text_key: &PyString,
    model: &PyString,
    total_jobs: usize,
    delim: i32,
    lowercase: bool,
    remove_punc: bool,
    show_progress: usize,
) -> PyResult<()> {
    let _ = multithread_tokenize_jsonl(
        text_path.to_string(),
        out_path.to_string(),
        text_key.to_string(),
        model.to_string(),
        total_jobs,
        delim,
        lowercase,
        remove_punc,
        show_progress,
    );
    Ok(())
}

#[pyfunction]
pub fn search_index(index: &PyString, val: u64) -> PyResult<(u64, u64)> {
    let mut sample_index = SampleIndex::new(index.to_string());
    Ok(sample_index.search(val).unwrap())
}

#[pyfunction]
pub fn get_part_offsets_py(
    total_len: usize,
    overlap: usize,
    total_jobs: usize,
) -> Vec<(usize, usize)> {
    get_part_offsets(total_len, overlap, total_jobs)
}

#[pyfunction]
pub fn find_tokenizer_boundaries_py(
    text_path: &PyString,
    total_jobs: usize,
) -> Vec<(usize, usize)> {
    find_tokenizer_boundaries(&text_path.to_string(), total_jobs)
}

#[pymodule]
pub fn suffix_arrays(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(make_part, m)?)?;
    m.add_function(wrap_pyfunction!(in_memory_merge, m)?)?;
    m.add_function(wrap_pyfunction!(get_part_offsets_py, m)?)?;
    m.add_function(wrap_pyfunction!(find_tokenizer_boundaries_py, m)?)?;
    // m.add_function(wrap_pyfunction!(make_part_in_memory, m)?)?;
    m.add_function(wrap_pyfunction!(build_in_memory_array, m)?)?;
    m.add_function(wrap_pyfunction!(find_duplicates, m)?)?;
    m.add_function(wrap_pyfunction!(collect, m)?)?;
    m.add_function(wrap_pyfunction!(get_idx, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_jsonl, m)?)?;
    m.add_function(wrap_pyfunction!(build_index_filtered_array_from_disk, m)?)?;
    m.add_function(wrap_pyfunction!(merge_from_disk, m)?)?;
    m.add_function(wrap_pyfunction!(search_index, m)?)?;
    m.add_function(wrap_pyfunction!(find, m)?)?;
    m.add_function(wrap_pyfunction!(find_off_disk, m)?)?;
    m.add_function(wrap_pyfunction!(find_multi, m)?)?;
    Ok(())
}
