use crate::table;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::iter::zip;

use crate::{
    get_next_maybe_skip, get_part_offsets, get_pointer, in_memory_position, to_bytes_index_only,
};

fn make_part_in_memory(text: &[u8], offset: u64, exclude: &HashSet<u64>) -> (Vec<u8>, usize) {
    let st = table::SuffixTable::new(text);
    let table = st.table();
    let ratio = ((text.len() as f64).log2() / 8.0).ceil() as usize;
    (to_bytes_index_only(table, ratio, 2, offset, exclude), ratio)
}

pub fn build_in_memory_array_impl(
    text: &[u8],
    num_parts: usize,
    num_threads: usize,
    hacksize: usize,
    exclude: &HashSet<u64>,
) -> (Vec<u8>, usize) {
    let job_params = get_part_offsets(text.len(), hacksize, num_parts);
    let mut parts: Vec<Vec<u8>> = vec![Vec::new(); num_parts];
    let mut ratios = vec![0; num_parts];

    let tables: Vec<Vec<u8>> = crossbeam::scope(|scope| {
        let mut results = Vec::new();
        for (i, (start, end)) in job_params.into_iter().enumerate() {
            let chunk = &text[start..end];
            let partial_table = scope.spawn(move |_| {
                let (tbl, r) = make_part_in_memory(chunk, start as u64, exclude);
                (tbl, r)
            });
            results.push(partial_table);
            parts[i] = chunk.to_vec();
        }
        results
            .into_iter()
            .enumerate()
            .map(|(i, res)| {
                let (tbl, r) = res.join().unwrap();
                ratios[i] = r;
                tbl
            })
            .collect()
    })
    .unwrap();
    println!("test: {:?}", ratios);
    in_memory_merge_impl(parts, tables, ratios, num_threads as i64, hacksize)
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct MergeState<'a> {
    suffix: &'a [u8],
    position: u64,
    table_index: usize,
}

impl<'a> Ord for MergeState<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.suffix.cmp(&self.suffix)
    }
}

impl<'a> PartialOrd for MergeState<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn in_memory_merge_impl(
    texts: Vec<Vec<u8>>,
    tables: Vec<Vec<u8>>,
    ratios: Vec<usize>,
    num_threads: i64,
    hacksize: usize,
) -> (Vec<u8>, usize) {
    // const hacksize:usize=100_000;

    let nn: usize = texts.len();

    let texts_len: Vec<usize> = texts
        .iter()
        .enumerate()
        .map(|(i, x)| x.len() - (if i + 1 == texts.len() { 0 } else { hacksize }))
        .collect();

    // let metadatas:Vec<usize> = (0..nn).map(|x| {
    //     // println!("{},{}: table: {}, text: {},", index_only, x, tables[x].len(), texts[x].len());
    //     assert!(tables[x].len()%texts[x].len() == 0);
    //     return tables[x].len();
    // }).collect();

    let big_ratio = ((texts_len.iter().sum::<usize>() as f64).log2() / 8.0).ceil() as usize;
    println!("Big Ratio: {}", big_ratio);

    // let ratio = metadatas[0] / (texts[0].len());
    // println!("little ratio: {}", ratio);
    // let ratios: Vec<usize> = metadatas.iter().zip(&texts).into_iter().map(|(m, t)| m/t.len()).collect();

    fn worker(
        texts: &Vec<Vec<u8>>,
        tables: &Vec<Vec<u8>>,
        starts: Vec<usize>,
        ends: Vec<usize>,
        texts_len: Vec<usize>,
        _part: usize,
        ratios: Vec<usize>,
        big_ratio: usize,
        hacksize: usize,
    ) -> Vec<u8> {
        let nn = texts.len();
        let mut idxs: Vec<u64> = starts.iter().map(|&x| x as u64).collect();
        let mut table_offsets: Vec<usize> = starts
            .iter()
            .enumerate()
            .map(|(i, &x)| x * &ratios[i])
            .collect(); //table_offsets

        let delta: Vec<u64> = (0..nn)
            .map(|x| {
                let pref: Vec<u64> = texts[..x].iter().map(|y| y.len() as u64).collect(); // lengths of first x texts
                pref.iter().sum::<u64>() - (hacksize * x) as u64 // total length of first x texts, without double counting hacksize overlaps == delta[x]
            })
            .collect();

        let mut next_table_mem: Vec<u8> = Vec::with_capacity(
            zip(&starts, &ends)
                .into_iter()
                .map(|(s, e)| (e - s) * big_ratio)
                .sum(),
        );

        let mut heap = BinaryHeap::new();

        // initialize heap with first positions (pointers) from tables
        for x in 0..nn {
            let position = get_next_maybe_skip(
                &tables[x],
                &mut table_offsets[x],
                ratios[x],
                &mut idxs[x],
                texts_len[x],
            );
            heap.push(MergeState {
                suffix: &texts[x][position as usize..],
                position: position,
                table_index: x,
            });
        }

        // while heap is non empty, pop off, write to table, advance corresponding tablestream and add new state to heap
        while let Some(MergeState {
            suffix: _suffix,
            position,
            table_index,
        }) = heap.pop()
        {
            // writes single pointer to next_table, taking into account we're indexing into the table_index'th text
            let next_ptr = &(position + delta[table_index] as u64).to_le_bytes()[..big_ratio];
            next_table_mem.extend_from_slice(next_ptr);

            let position = get_next_maybe_skip(
                &tables[table_index],
                &mut table_offsets[table_index],
                ratios[table_index],
                &mut idxs[table_index],
                texts_len[table_index],
            );
            if position == u64::MAX {
                continue;
            }

            //
            if idxs[table_index] <= ends[table_index] as u64 {
                heap.push(MergeState {
                    suffix: &texts[table_index][position as usize..],
                    position: position,
                    table_index: table_index,
                });
            }
            // prev = next;
        }
        next_table_mem
    }

    // Make sure we have enough space to take strided offsets for multiple threads
    // This should be an over-approximation, and starts allowing new threads at 1k of data
    let num_threads = std::cmp::min(
        num_threads,
        std::cmp::max((texts[0].len() as i64 - 1024) / 10, 1),
    );
    println!(
        "AA {} {} {}",
        num_threads,
        texts[0].len(),
        (texts[0].len() as i64 - 1024) / 10
    );

    // Start a bunch of jobs that each work on non-overlapping regions of the final resulting suffix array
    // Each job is going to look at all of the partial suffix arrays to take the relavent slice.

    let answer: Vec<Vec<u8>> = crossbeam::scope(|scope| {
        let mut starts = vec![0; nn];
        let mut results = Vec::new();
        for i in 0..num_threads as usize {
            let texts = &texts;
            let tables = &tables;
            let mut ends: Vec<usize> = vec![0; nn];
            if i < num_threads as usize - 1 {
                ends[0] = (tables[0].len() / ratios[0] + (num_threads as usize))
                    / (num_threads as usize)
                    * (i + 1);

                let ptr = get_pointer(&tables[0], ends[0] * ratios[0], ratios[0]) as usize;
                let end_seq = &texts[0][ptr..];

                for j in 1..ends.len() {
                    ends[j] = in_memory_position(&texts[j], &tables[j], end_seq, ratios[j]);
                }
            } else {
                for j in 0..ends.len() {
                    ends[j] = tables[j].len() / ratios[j];
                }
            }

            let starts2 = starts.clone();
            let ends2 = ends.clone();
            let texts_len2 = texts_len.clone();
            let ratios2 = ratios.clone();
            let one_result = scope.spawn(move |_| {
                // println!("Starting merge worker {}", i);
                worker(
                    texts,
                    tables,
                    starts2,
                    ends2,
                    texts_len2,
                    i,
                    ratios2,
                    big_ratio as usize,
                    hacksize,
                )
            });

            results.push(one_result);

            for j in 0..ends.len() {
                starts[j] = ends[j];
            }
        }
        results.into_iter().map(|res| res.join().unwrap()).collect()
    })
    .unwrap();

    (answer.into_iter().flatten().collect(), big_ratio)
}
