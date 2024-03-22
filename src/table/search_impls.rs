use crate::off_disk_utils::*;
use crate::tokenize::*;
use crate::{from_bytes, get_pointer, in_memory_position, next_pointer_skip, to_bytes};
use std::fs;
use std::io::prelude::*;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

fn find_max_match_length_sg(
    text1: &[u8],
    text2: &[u8],
    locations1: &[u8], // start byte of matches from text1
    locations2: &[u8],
    width1: usize,
    width2: usize,
    length_threshold: usize,
    skipgram_budget: u64,
    eos_boundary: [u8; 2],
) -> (
    Vec<u64>, // max lengths
    Vec<u64>, // locs (in text2) with max length)
    Vec<u64>, // skips used
) {
    let true_locations1 = from_bytes(locations1.to_vec(), width1);
    let true_locations2 = from_bytes(locations2.to_vec(), width2);
    let mut max_lengths = Vec::new();
    let mut locations = Vec::new();
    let mut skips = Vec::new();

    for loc1 in true_locations1.iter() {
        let start1 = *loc1 as usize;
        let end1 = std::cmp::min((loc1 + 4096) as usize, text1.len());
        let targ1 = &text1[start1..end1];
        let mut max_len = length_threshold;
        let mut max_loc = true_locations2[0] as u64;
        let mut skipped = 0u64;
        for loc2 in true_locations2.iter() {
            let start2 = *loc2 as usize;
            let end2 = std::cmp::min((loc2 + 4096) as usize, text2.len());
            let targ2 = &text2[start2..end2];
            // let mut curr_match_len = length_threshold;
            let max_possible_len = std::cmp::min(end1 - start1, end2 - start2);
            let mut skips_taken = 0u64;
            if let Some(mut curr_match_len) = (length_threshold..max_possible_len - 2)
                .filter(|x| x % 2 == 0)
                .find(|&l| {
                    let mismatch = !(targ1[l..l + 2] == targ2[l..l + 2]);
                    skips_taken += mismatch as u64;
                    ((mismatch) && (skips_taken >= skipgram_budget))
                        || (targ1[l..l + 2] == eos_boundary)
                        || (l + 2 >= max_possible_len)
                })
            {
                while !(targ1[curr_match_len - 2..curr_match_len]
                    == targ2[curr_match_len - 2..curr_match_len])
                {
                    curr_match_len -= 2;
                }
                if curr_match_len > max_len {
                    max_len = curr_match_len;
                    max_loc = *loc2;
                    skipped = skips_taken
                }
            } else {
                max_len = 4096;
                max_loc = *loc2;
            }

            if max_len == 4096 {
                break;
            };
        }
        max_lengths.push(max_len as u64);
        locations.push(max_loc);
        skips.push(skipped);
    }
    (max_lengths, locations, skips)
}

fn find_max_match_length(
    text1: &[u8],
    text2: &[u8],
    locations1: &[u8], // start byte of matches from text1
    locations2: &[u8],
    width1: usize,
    width2: usize,
    length_threshold: usize,
) -> (
    Vec<u64>, // max lengths
    Vec<u64>, // locs (in text2) with max length
    Vec<u64>, // dummy vec
) {
    let true_locations1 = from_bytes(locations1.to_vec(), width1);
    let true_locations2 = from_bytes(locations2.to_vec(), width2);
    let mut max_lengths = Vec::new();
    let mut locations = Vec::new();
    for loc1 in true_locations1.iter() {
        let start1 = *loc1 as usize;
        let end1 = std::cmp::min((loc1 + 4096) as usize, text1.len());
        let targ1 = &text1[start1..end1];
        let mut max_len = length_threshold;
        let mut max_loc = true_locations2[0] as u64;
        for loc2 in true_locations2.iter() {
            let start2 = *loc2 as usize;
            let end2 = std::cmp::min((loc2 + 4096) as usize, text2.len());
            let targ2 = &text2[start2..end2];
            // let mut curr_match_len = length_threshold;
            let max_possible_len = std::cmp::min(end1 - start1, end2 - start2);
            if let Some(curr_match_len) = (length_threshold..max_possible_len - 2)
                .filter(|x| x % 2 == 0)
                // Note we don't need to check eos boundary as we use different boundary tokens for each text,
                //so matches never cross these boundaries without skipgram
                .find(|&l| !(targ1[l..l + 2] == targ2[l..l + 2]) || (l + 2 >= max_possible_len))
            {
                if curr_match_len > max_len {
                    max_len = curr_match_len;
                    max_loc = *loc2;
                }
            } else {
                max_len = 4096;
                max_loc = *loc2;
            }

            if max_len == 4096 {
                break;
            };
        }
        max_lengths.push(max_len as u64);
        locations.push(max_loc);
    }
    (max_lengths, locations, vec![0u64])
}

pub fn find_duplicates_impl(
    text1: &[u8],
    table1: &[u8],
    text2: &[u8],
    table2: &[u8],
    ratio1: usize,
    ratio2: usize,
    length_threshold: usize,
    num_threads: i64,
    find_max: bool,
    skipgram_budget: usize,
    eos: u8,
) -> Vec<Vec<u8>> {
    let text_len1 = text1.len();
    let table_len1 = table1.len();
    let text_len2 = text2.len();
    let table_len2 = table2.len();
    if (text_len1 == 0) || (table_len1 == 0) || (text_len2 == 0) || (table_len2 == 0) {
        // quick exit on errors
        return vec![vec![0], vec![0], vec![0], vec![0], vec![0], vec![0]];
    }

    fn worker(
        text1: &[u8],
        text2: &[u8],
        start1: usize,
        end1: usize,
        start2: usize,
        end2: usize,
        table1: &[u8],
        table2: &[u8],
        length_threshold: usize,
        size_width_1: usize,
        size_width_2: usize,
        find_max: bool,
        skipgram_budget: usize,
        eos: u8,
        // worker_num: i64,
        // log_freq: i32,
    ) -> Vec<Vec<u8>> {
        let mut offset1 = start1 * size_width_1;
        let mut offset2 = start2 * size_width_2;

        let (mut location1, skip1) = next_pointer_skip(table1, &mut offset1, size_width_1);

        let (mut location2, skip2) = next_pointer_skip(table2, &mut offset2, size_width_2);

        let mut dupes1: Vec<u8> = Vec::new();
        let mut sizes1 = Vec::new();
        let mut dupes2: Vec<u8> = Vec::new();
        let mut sizes2 = Vec::new();
        let mut max_lengths = Vec::new();
        let mut max_locations = Vec::new();
        let mut skipped = Vec::new();

        let mut i = start1 + skip1 - 1;
        let mut j = start2 + skip2 - 1;
        let mut skips: usize;
        // let mut its = 0;

        while i < end1 && j < end2 {
            let mut suf1 = &text1[location1 as usize..];
            let mut suf2 = &text2[location2 as usize..];

            // Do we have a match between the suffix that begins at location1 in text1
            // and the suffix that begins at location2 in text2?
            // To check this we need (a) both are long enough, and
            // (b) the match is of length at least length_threshold

            let does_match = suf1.len() >= length_threshold
                && suf2.len() >= length_threshold
                && suf1[..length_threshold] == suf2[..length_threshold];

            if does_match {
                // We have a match between a subsequence in text1 and text2
                let target_suf = &suf1[..length_threshold]; // wlog. equals suf2[..length_threshold]

                // We want the matches to be clustered, so let's find all matches from
                // the first string that are equal to target_suf
                let mut matches1 = Vec::new();
                let mut matches2 = Vec::new();

                // let start = i;
                let mut match_count = 0u64;
                while suf1.len() >= length_threshold && &suf1[..length_threshold] == target_suf {
                    matches1
                        .extend_from_slice(&to_bytes(&[location1 as u64][..], size_width_1)[..]);
                    (location1, skips) = next_pointer_skip(table1, &mut offset1, size_width_1);
                    i += skips;
                    match_count += 1;
                    if location1 == std::u64::MAX {
                        break;
                    }
                    suf1 = &text1[location1 as usize..];
                }
                sizes1.extend_from_slice(&to_bytes(&[match_count][..], size_width_1)[..]);

                // let start = j;
                let mut match_count = 0u64;
                while suf2.len() >= length_threshold && &suf2[..length_threshold] == target_suf {
                    matches2
                        .extend_from_slice(&to_bytes(&[location2 as u64][..], size_width_2)[..]);
                    (location2, skips) = next_pointer_skip(table2, &mut offset2, size_width_2);
                    j += skips;
                    match_count += 1;
                    if location2 == std::u64::MAX {
                        break;
                    }
                    suf2 = &text2[location2 as usize..];
                }

                sizes2.extend_from_slice(&to_bytes(&[match_count][..], size_width_2)[..]);

                if find_max {
                    // How long is the current match?
                    let sgb = skipgram_budget as u64;
                    let (max_matches, max_locs, skipped_counts) = if sgb == 0 {
                        find_max_match_length(
                            text1,
                            text2,
                            &matches1[..],
                            &matches2[..],
                            size_width_1,
                            size_width_2,
                            length_threshold,
                        )
                    } else {
                        find_max_match_length_sg(
                            text1,
                            text2,
                            &matches1[..],
                            &matches2[..],
                            size_width_1,
                            size_width_2,
                            length_threshold,
                            sgb,
                            [eos, 0u8],
                        )
                    };
                    max_lengths.extend_from_slice(&to_bytes(&max_matches[..], size_width_1));
                    max_locations.extend_from_slice(&to_bytes(&max_locs[..], size_width_2));
                    skipped.extend_from_slice(&to_bytes(&skipped_counts[..], size_width_1));
                }
                dupes1.extend_from_slice(&matches1[..]);
                dupes2.extend_from_slice(&matches2[..]);
            } else if suf1 < suf2 {
                // No match, and the first suffix is smaller. Increment the smaller one
                (location1, skips) = next_pointer_skip(table1, &mut offset1, size_width_1);
                i += skips;
            } else if suf2 < suf1 {
                // No match, and the second suffix is smaller. Increment the smaller one
                (location2, skips) = next_pointer_skip(table2, &mut offset2, size_width_2);
                j += skips;
            } else {
                // This happens only when
                // 1. The two suffixes are identical
                // 2. But they're not yet long enough for it to "count"
                // so we just increment one of the poitners WLOG
                assert!(&suf1 == &suf2);
                assert!(suf1.len() < 100 || suf2.len() < 100);
                (location1, skips) = next_pointer_skip(table1, &mut offset1, size_width_1);
                i += skips;
            }
            if (location2 == std::u64::MAX) || (location1 == std::u64::MAX) {
                break;
            }
        }

        return vec![
            dupes1,
            sizes1,
            dupes2,
            sizes2,
            max_lengths,
            max_locations,
            skipped,
        ];
    }

    // Start a bunch of jobs that each work on non-overlapping regions of the suffix array.
    // We use text_len here as we index into tables via ratios
    // Note to self: this will be an issue when we try to index into filtered tables
    let increment: i64 = ((table_len1) as i64 - num_threads) / num_threads;
    let answer = crossbeam::scope(|scope| {
        let mut result = Vec::with_capacity(num_threads as usize);
        // let text1 = &text1[..];
        // let text2 = &text2[..];
        // let table1 = &table1[..];
        // let table2 = &table2[..];
        let mut last_end = 0;
        for i in 0..num_threads {
            // starting points for this thread for text1/table1
            let a = std::cmp::max(0i64, i * increment - 1) as usize;
            let b = std::cmp::min(((i + 1) * increment - 1) as usize, table1.len() / ratio1);
            println!("This is b: {}", b);
            let this_start = last_end;
            let end_ptr = get_pointer(table1, b * ratio1, ratio1 as usize) as usize;
            let end_seq = &text1[end_ptr..];
            let this_end = in_memory_position(text2, table2, end_seq, ratio2);

            last_end = this_end;
            let one_result = scope.spawn(move |_| {
                return worker(
                    text1,
                    text2,
                    a,          //start1
                    b,          //end1
                    this_start, //start2
                    this_end,   //end2
                    table1,
                    table2,
                    length_threshold,
                    ratio1 as usize,
                    ratio2 as usize,
                    find_max,
                    skipgram_budget,
                    eos,
                    // i,
                    // 1000
                );
            });
            result.push(one_result);
        }
        // println!("All workers returned");
        let r = result.into_iter().map(|t| t.join().unwrap()).fold(
            vec![Vec::<u8>::new(); 7],
            |mut acc, v| {
                acc[0].extend_from_slice(&v[0][..]); //pointer to dupes in text1
                acc[1].extend_from_slice(&v[1][..]); //number of said dupes in text1
                acc[2].extend_from_slice(&v[2][..]); //pointers text2
                acc[3].extend_from_slice(&v[3][..]); //number of dupes text2
                acc[4].extend_from_slice(&v[4][..]); //length of max match of text1 dupe
                acc[5].extend_from_slice(&v[5][..]); //location of match (dupe at)
                acc[6].extend_from_slice(&v[6][..]); //number of skipped tokens per match
                acc
            },
        );
        r
    });
    answer.unwrap()
}

pub fn find_off_disk_impl(query: &String, tokens: &String, tokenizer: &String) -> (u64, Vec<u64>) {
    let query: Vec<u8> = tokenizer_encode(tokenizer, query)
        .unwrap()
        .iter()
        .map(|token| (*token as u16).to_le_bytes())
        .flatten()
        .collect();
    let tokens = tokens.to_string();
    let suffix_array = format!("{}.st", &tokens);

    let array = filebuffer::FileBuffer::open(suffix_array).unwrap();
    let tokens = filebuffer::FileBuffer::open(tokens).unwrap();
    count_occurances(&tokens, &array, &query[..])
}

/*
A near-duplicate of Carlini's across-similar
*/
pub fn find_off_disk_file_impl(
    query_file: &String,
    tokens: &String,
    cache_dir: &String,
    length_threshold: usize,
    num_threads: i64,
    linear_search: bool,
) -> () {
    let query_tokens = filebuffer::FileBuffer::open(query_file).unwrap();
    let search_tokens = filebuffer::FileBuffer::open(tokens).unwrap();
    let (query_ratio, search_ratio, query_size, search_size) = {
        let query_array = filebuffer::FileBuffer::open(format!("{}.st", query_file)).unwrap();
        let search_array = filebuffer::FileBuffer::open(format!("{}.st", tokens)).unwrap();
        (
            *query_array.last().unwrap(),
            *search_array.last().unwrap(),
            query_array.len() - 1,
            search_array.len() - 1,
        ) // The last byte of the suffix arrays is the width
    };

    fn worker(
        text1: &[u8],
        text2: &[u8],
        start1: usize,
        end1: usize,
        start2: usize,
        end2: usize,
        data_file_1: String,
        data_file_2: String,
        cache_dir: String,
        length_threshold: usize,
        size_width_1: usize,
        size_width_2: usize,
        pb: Option<ProgressBar>,
        linear_search: bool,
    ) -> usize {
        let mut table1 = TableStream::new(format!("{data_file_1}"), size_width_1, linear_search);
        let mut table2 = TableStream::new(format!("{data_file_2}"), size_width_2, linear_search);
        let _ = table1.seek(start1 as u64);
        let _ = table2.seek(start2 as u64);

        let mut location1 = table1.get_index();
        let mut location2 = table2.get_index();
        // What do you mean this looks ugly. I see no problem here!
        let mut outfile1 = std::io::BufWriter::new(
            fs::File::create(format!(
                "{}/dups_{}_{}-{}_{}_{}-{}",
                cache_dir,
                data_file_1.split("/").last().unwrap(),
                start1,
                end1,
                data_file_2.split("/").last().unwrap(),
                start2,
                end2,
            ))
            .unwrap(),
        );
        let mut outfile1_sizes = std::io::BufWriter::new(
            fs::File::create(format!(
                "{}/sizes_{}_{}-{}_{}_{}-{}",
                cache_dir,
                data_file_1.split("/").last().unwrap(),
                start1,
                end1,
                data_file_2.split("/").last().unwrap(),
                start2,
                end2,
            ))
            .unwrap(),
        );

        let mut outfile2 = std::io::BufWriter::new(
            fs::File::create(format!(
                "{}/dups_{}_{}-{}_{}_{}-{}",
                cache_dir,
                data_file_2.split("/").last().unwrap(),
                start2,
                end2,
                data_file_1.split("/").last().unwrap(),
                start1,
                end1,
            ))
            .unwrap(),
        );
        let mut outfile2_sizes = std::io::BufWriter::new(
            fs::File::create(format!(
                "{}/sizes_{}_{}-{}_{}_{}-{}",
                cache_dir,
                data_file_2.split("/").last().unwrap(),
                start2,
                end2,
                data_file_1.split("/").last().unwrap(),
                start1,
                end1,
            ))
            .unwrap(),
        );

        let mut duplicate_count = 0;
        let mut i = start1;
        let mut j = start2;
        let mut its = 0;

        while i < end1 && j < end2 {
            // if its%1000 == 0 { println!("thread {}, its: {}: {} / {}; {} / {} ",thread_no, its,  i, end1,j,end2 ); }

            let mut suf1 = &text1[location1 as usize..];
            let mut suf2 = &text2[location2 as usize..];

            // Do we have a match between the suffix that begins at location1 in text1
            // and the suffix that begins at location2 in text2?
            // To check this we need (a) both are long enough, and
            // (b) the match is of length at least length_threshold

            let does_match = suf1.len() >= length_threshold
                && suf2.len() >= length_threshold
                && suf1[..length_threshold] == suf2[..length_threshold];

            if does_match {
                // We have a match between a subsequence in text1 and text2
                let target_suf = &suf1[..length_threshold]; // wlog. equals suf2[..length_threshold]

                // We want the matches to be clustered, so let's find all matches from
                // the first string that are equal to target_suf
                let start = i;
                while suf1.len() >= length_threshold && &suf1[..length_threshold] == target_suf {
                    outfile1
                        .write_all(&to_bytes(&[location1 as u64][..], size_width_1)[..])
                        .expect("Ok");

                    location1 = table1.get_next_index();
                    i += 1;
                    if location1 == std::u64::MAX {
                        break;
                    }
                    suf1 = &text1[location1 as usize..];
                }
                duplicate_count += i - start;
                outfile1_sizes
                    .write_all(&to_bytes(&[(i - start) as u64][..], size_width_1)[..])
                    .expect("Ok");

                // And now find all matches from the second string that are equal to target_suf
                let start = j;
                while suf2.len() >= length_threshold && &suf2[..length_threshold] == target_suf {
                    outfile2
                        .write_all(&to_bytes(&[location2 as u64][..], size_width_2)[..])
                        .expect("Ok");

                    location2 = table2.get_next_index();
                    j += 1;
                    if location2 == std::u64::MAX {
                        break;
                    }
                    suf2 = &text2[location2 as usize..];
                }
                duplicate_count += j - start;
                outfile2_sizes
                    .write_all(&to_bytes(&[(j - start) as u64][..], size_width_2)[..])
                    .expect("Ok");
            } else if suf1 < suf2 {
                // No match, and the first suffix is smaller. Increment the smaller one
                i += 1;

                location1 = table1.get_next_index();
            } else if suf2 < suf1 {
                // No match, and the second suffix is smaller. Increment the smaller one
                if linear_search {
                    j += 1;
                    location2 = table2.get_next_index();
                    if pb.is_some() {
                        pb.as_ref().unwrap().inc(1);
                    };
                } else {
                    let jump_to =
                        table2.binary_search_position(suf1, j, end2 as usize, length_threshold);
                    // println!("bs: {} {} {}", j, end2, jump_to);
                    location2 = table2.get_index_at(jump_to);
                    if pb.is_some() {
                        pb.as_ref().unwrap().inc(jump_to - j as u64)
                    };
                    j = jump_to as usize;
                }
            } else {
                // This happens only when
                // 1. The two suffixes are identical
                // 2. But they're not yet long enough for it to "count"
                // so we just increment one of the poitners WLOG

                assert!(&suf1 == &suf2);
                assert!(suf1.len() < 100 || suf2.len() < 100);
                i += 1;
                location1 = table1.get_next_index();
            }
            its += 1;
        }
        if pb.is_some() {
            pb.as_ref().unwrap().finish()
        };

        return duplicate_count;
    }

    // Start a bunch of jobs that each work on non-overlapping regions of the suffix array.
    let increment: i64 = ((query_size / query_ratio as usize) as i64 - num_threads) / num_threads;
    println!("Increment: {increment}");
    let m = MultiProgress::new();
    let _answer = crossbeam::scope(|scope| {
        let mut result = Vec::with_capacity(num_threads as usize);
        let text1 = &query_tokens;
        let text2 = &search_tokens;
        let mut last_end = 0;
        for i in 0..num_threads {
            let a = std::cmp::max(0i64, i * increment - 1) as usize;
            let b = std::cmp::min(
                ((i + 1) * increment) as usize,
                query_size / query_ratio as usize,
            );

            let mut table1 =
                std::io::BufReader::new(fs::File::open(format!("{}.st", query_file)).unwrap());
            let mut table2 =
                std::io::BufReader::new(fs::File::open(format!("{}.st", tokens)).unwrap());
            let this_start = last_end;
            // println!("i {i}, b {b}, max is {}", query_size/(query_ratio as usize));
            let end_seq = &text1[table_load_disk(&mut table1, b, query_ratio as usize)..];
            let this_end = off_disk_position(
                text2,
                &mut table2,
                end_seq,
                search_size / (search_ratio as usize),
                search_ratio as usize,
            );

            last_end = this_end;
            // println!("start {} {}", this_start, this_end);
            let pb: Option<ProgressBar> = match i {
                0 | 1 | 2 | 3 => {
                    let style = ("Vertical: ", "█▇▆▅▄▃▂▁  ", "green");
                    let bar = m.add(ProgressBar::new((this_end - this_start) as u64));
                    bar.set_prefix(format!("Thread {i:0>3}:"));

                    bar.set_style(ProgressStyle::with_template(&format!(
                        "{{prefix:.bold}} [{{elapsed_precise}} ({{per_sec}})]▕{{bar:.{}}}▏{{percent}}% {{pos:>9}}/{{len:>9}}",
                        style.2,
                    ))
                    .unwrap()
                    .progress_chars(style.1),);
                    Some(bar)
                }
                _ => None,
            };
            let one_result = scope.spawn(move |_| {
                return worker(
                    text1,
                    text2,
                    a,
                    b,
                    this_start,
                    this_end,
                    query_file.clone(),
                    tokens.clone(),
                    cache_dir.clone(),
                    length_threshold,
                    query_ratio as usize,
                    search_ratio as usize,
                    pb,
                    linear_search,
                );
            });
            result.push(one_result);
        }

        let thread_sum: usize = result.into_iter().map(|t| t.join().unwrap()).sum();
        println!("Duplicates found: {:?}", thread_sum);
    });
    ()
}
