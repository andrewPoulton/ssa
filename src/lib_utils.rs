use crate::table_utils::*;
use std::collections::HashSet;

pub fn to_bytes(input: &[u64], size_width: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size_width * input.len());

    for value in input {
        bytes.extend(&value.to_le_bytes()[..size_width]);
    }
    bytes
}

pub fn to_bytes_index_only(
    input: &[u64],
    size_width: usize,
    index_only: usize,
    offset: u64,
    exclude: &HashSet<u64>,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size_width * input.len());
    let mut filtered = 0;
    for value in input {
        if (value % index_only as u64 == 0) && (!exclude.contains(&(value + offset))) {
            bytes.extend(&value.to_le_bytes()[..size_width]);
        } else {
            filtered += 1;
        }
    }
    println!(
        "filtered {}. Before: {}, after: {}",
        filtered,
        input.len(),
        bytes.len() / size_width
    );
    bytes
}

/* Convert a uint8 array to a uint64. Only called on (relatively) small files. */
pub fn from_bytes(input: Vec<u8>, size_width: usize) -> Vec<u64> {
    // println!("S {}", input.len());
    assert!(input.len() % size_width == 0);
    let mut bytes: Vec<u64> = Vec::with_capacity(input.len() / size_width);

    let mut tmp = [0u8; 8];
    // todo learn rust macros, hope they're half as good as lisp marcos
    // and if they are then come back and optimize this
    for i in 0..input.len() / size_width {
        tmp[..size_width].copy_from_slice(&input[i * size_width..i * size_width + size_width]);
        bytes.push(u64::from_le_bytes(tmp));
    }

    bytes
}

pub fn get_part_offsets(
    total_len: usize,
    overlap: usize,
    total_jobs: usize,
) -> Vec<(usize, usize)> {
    // assert!(total_len % index_only == 0)
    let mut offsets = Vec::with_capacity(total_jobs);
    let mut bytes_per_job = total_len / total_jobs;
    bytes_per_job -= bytes_per_job % 2;
    for i in 0..total_jobs {
        let start = i * bytes_per_job;
        let end = std::cmp::min((i + 1) * bytes_per_job + overlap, total_len);
        offsets.push((start, end));
        if end == total_len {
            break;
        }
    }
    offsets[total_jobs - 1].1 = total_len;
    offsets
}
pub fn index_to_hashset(index_bytes: Vec<u8>) -> HashSet<u64> {
    let width = estimate_ratio(&index_bytes);
    let mut index_set = HashSet::new();
    if width == 0 {
        return index_set;
    };
    // index_bytes.chunks_exact(width)
    index_set.extend(index_bytes.chunks_exact(width).map(|bytes| {
        let mut tmp = [0u8; 8];
        tmp[..width].copy_from_slice(bytes);
        u64::from_le_bytes(tmp)
    }));

    index_set
}
