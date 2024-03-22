use filebuffer::FileBuffer;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

use crate::from_bytes;

/*
 * We're going to work with suffix arrays that are on disk, and we often want
 * to stream them top-to-bottom. This is a datastructure that helps us do that:
 * we read 1MB chunks of data at a time into the cache, and then fetch new data
 * when we reach the end.
 */
pub struct TableStream {
    // TODO: make fully-featured. have fp to tokens, binary search for seqs
    tokens: FileBuffer,
    array: BufReader<File>,
    cache: [u8; 8],
    size_width: usize,
    pub array_length: u64,
    pub tokens_length: u64,
    pub tell: u64,
}

impl TableStream {
    pub fn new(path: String, size_width: usize, linear_search_optimized: bool) -> Self {
        let array_path = format!("{path}.st");
        let array_length = std::fs::metadata(&array_path).unwrap().len();
        let token_length = std::fs::metadata(&path).unwrap().len();
        let capacity = if linear_search_optimized {
            1024 * 1024
        } else {
            size_width * 8
        };
        let table = TableStream {
            tokens: FileBuffer::open(&path).unwrap(),
            array: std::io::BufReader::with_capacity(
                capacity,
                fs::File::open(&array_path).unwrap(),
            ),
            cache: [0u8; 8],
            size_width: size_width,
            array_length: (array_length - 1) / size_width as u64,
            tokens_length: token_length,
            tell: 0,
        };
        // table.seek (offset ).expect ("Seek failed!");
        table
    }

    pub fn get_suffix(&self) -> std::io::Result<&[u8]> {
        Ok(&self.tokens[self.get_index() as usize..])
    }

    pub fn get_suffix_at(&mut self, pos: u64) -> std::io::Result<&[u8]> {
        self.seek(pos)?;
        self.get_suffix()
    }

    pub fn get_index(&self) -> u64 {
        u64::from_le_bytes(self.cache)
    }

    pub fn get_index_at(&mut self, pos: u64) -> u64 {
        self.seek(pos);
        self.get_index()
    }

    pub fn read(&mut self) -> () {
        self.array.read_exact(&mut self.cache[..self.size_width]);
        self.tell += self.size_width as u64;
    }

    pub fn get_next_index(&mut self) -> u64 {
        self.read();
        self.get_index()
    }

    pub fn seek(&mut self, pos: u64) -> std::io::Result<()> {
        self.tell = self
            .array
            .seek(std::io::SeekFrom::Start(pos * self.size_width as u64))?;
        self.read();
        Ok(())
    }

    pub fn binary_search_position(
        &mut self,
        query: &[u8],
        mut left: usize,
        mut right: usize,
        threshold: usize,
    ) -> u64 {
        // Get the insert location for query[..threshold]
        let qthresh = std::cmp::min(threshold, query.len());
        while left < right {
            let mid = (left + right) / 2;
            let targ = self.get_suffix_at(mid as u64).unwrap();
            let tthresh = std::cmp::min(threshold, targ.len());
            if query[..qthresh] <= targ[..tthresh] {
                if right == mid {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            } else {
                left = mid + 1;
            }
        }
        left as u64
    }
}

// impl Seek for TableStream {
//     fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
//         self.file.seek(pos)
//     }
// }

// /* Make a table from a file path and a given offset into the table */
// fn make_table(path: std::string::String,
//     offset: usize,
//     size_width: usize) -> TableStream {
// let mut table = TableStream {
// file: std::io::BufReader::with_capacity(1024*1024, fs::File::open(path).unwrap()),
// cache: [0u8; 8],
// size_width: size_width
// };
// return table;
// }

/* Get the next word from the suffix table. */
// pub fn get_next_pointer_from_table_canfail(tablestream: &mut TableStream) -> u64 {
//     let ok = tablestream
//         .array
//         .read_exact(&mut tablestream.cache[..tablestream.size_width]);
//     let bad = match ok {
//         Ok(_) => false,
//         Err(_) => true,
//     };
//     if bad {
//         return std::u64::MAX;
//     }
//     let out = u64::from_le_bytes(tablestream.cache);
//     return out;
// }

// pub fn get_next_pointer_from_table(tablestream: &mut TableStream) -> u64 {
//     let r = get_next_pointer_from_table_canfail(tablestream);
//     if r == std::u64::MAX {
//         panic!("Reached EOF badly");
//     }
//     return r;
// }

pub fn table_load_filebuffer(table: &filebuffer::FileBuffer, index: usize, width: usize) -> usize {
    let mut tmp = [0u8; 8];
    tmp[..width].copy_from_slice(&table[index * width..index * width + width]);
    return u64::from_le_bytes(tmp) as usize;
}

/* For a suffix array, just compute A[i], but load off disk because A is biiiiiiigggggg. */
pub fn table_load_disk(table: &mut BufReader<File>, index: usize, size_width: usize) -> usize {
    // println!("Index: {index}");
    table
        .seek(std::io::SeekFrom::Start((index * size_width) as u64))
        .expect("Seek failed!");
    let mut tmp = [0u8; 8];
    table
        .read_exact(&mut tmp[..size_width])
        .unwrap_or_else(|_| {
            println!(
                "Failed seeking to {index} with width {size_width} - aiming for position {}",
                index * size_width
            );
            panic!();
        });
    return u64::from_le_bytes(tmp) as usize;
}

// As above, but a slice
pub fn table_read_disk(
    table: &filebuffer::FileBuffer,
    start: usize,
    end: usize,
    size_width: usize,
) -> Vec<u64> {
    let buf = &table[start * size_width..end * size_width];
    from_bytes(buf.to_vec(), size_width)
}

/* Binary search to find where query happens to exist in text */
pub fn off_disk_position(
    text: &[u8],
    table: &mut BufReader<File>,
    query: &[u8],
    table_size: usize,
    size_width: usize,
) -> usize {
    let (mut left, mut right) = (0, table_size);
    while left < right {
        let mid = (left + right) / 2;
        if query < &text[table_load_disk(table, mid, size_width)..] {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    left
}

/*
 * Helper function to actually do the count of the number of times something is repeated.
 * This should be fairly simple.
 * First, perform binary search using the on-disk suffix array to find the first place
 * where the string occurrs. If it doesn't exist then return 0.
 * Then, binary search again to find the last location it occurrs.
 * Return the difference between the two.
 */
pub fn count_occurances(
    text: &filebuffer::FileBuffer,
    // size_text: u64,
    table: &filebuffer::FileBuffer,
    // size: u64,
    query: &[u8],
    // size_width: usize,
) -> (u64, Vec<u64>) {
    let mut buf: &[u8];
    // assert!(size % (size_width as u64) == 0);
    let length_threshold = query.len();
    let ratio = *table.last().unwrap() as usize;

    let mut low = 0;
    let mut high = (table.len() - 1) / ratio;
    let mut mid = 0;
    let mut it = 1;
    while low < high {
        mid = (high + low) / 2;
        let pos = table_load_filebuffer(&table, mid as usize, ratio);

        if pos + length_threshold < text.len() as usize {
            buf = &text[pos..pos + query.len()];
        } else {
            buf = &text[pos..];
        }
        // println!(
        //     "it {}: lo {}, mid {}, hi {}, pos {}, buf {:?}",
        //     it, low, mid, high, pos, buf
        // );

        if query <= buf {
            high = mid;
        } else {
            low = mid + 1;
        }
        it += 1;
    }
    let start = low;
    // println!("lo {}, mid {}, hi {}", low, mid, high);
    let pos = table_load_filebuffer(&table, low as usize, ratio);
    if pos + length_threshold < text.len() {
        buf = &text[pos..pos + query.len()];
    } else {
        buf = &text[pos..];
    }
    // println!("Final buffer: {:?}", &buf);
    if query != buf {
        return (0, Vec::new()); // not found
    }

    high = (table.len() - 1) / ratio;
    while low < high {
        let mid = (high + low) / 2;
        let pos = table_load_filebuffer(&table, mid as usize, ratio);

        if pos + length_threshold < text.len() {
            buf = &text[pos..pos + query.len()];
        } else {
            buf = &text[pos..];
        }

        if query != buf {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    let locs = table_read_disk(&table, start, low, ratio);
    let occurances = (low - start) as u64;
    (occurances, locs)
}

pub fn read_file_to_buffer(path: &str) -> Vec<u8> {
    let size = std::fs::metadata(path.to_string()).unwrap().len();
    let mut buffer: Vec<u8> = vec![0; size as usize];
    let mut reader = BufReader::new(fs::File::open(path.to_string()).unwrap());
    let _ = reader.read_exact(&mut buffer[..]);
    buffer
}
