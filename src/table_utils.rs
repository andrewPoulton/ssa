pub fn get_pointer(table: &[u8], offset: usize, size_width: usize) -> u64 {
    if offset + size_width > table.len() {
        // for compatibilty with off-disk reads
        // println!("Offset: {}, table_len: {}", offset, table.len());
        // println!(
        //     "Offset {} size_width {} table.len() {}",
        //     offset,
        //     size_width,
        //     table.len()
        // );
        return std::u64::MAX;
    }
    // let offset_ = offset * size_width;
    let mut tmp: [u8; 8] = [0u8; 8];
    tmp[..size_width].copy_from_slice(&table[offset..(offset + size_width)]);
    u64::from_le_bytes(tmp)
}

pub fn next_pointer(table: &[u8], offset: &mut usize, size_width: usize) -> u64 {
    let out = get_pointer(table, *offset, size_width);
    *offset += size_width;
    out
}

pub fn next_pointer_skip(table: &[u8], offset: &mut usize, size_width: usize) -> (u64, usize) {
    let mut ptr: u64;
    let mut skips = 1usize;
    loop {
        ptr = get_pointer(table, *offset, size_width);
        *offset += size_width;
        if (ptr % 2 == 0) || (ptr == std::u64::MAX) {
            break (ptr, skips);
        }
        skips += 1;
    }
}

pub fn get_next_maybe_skip(
    table: &Vec<u8>,
    offset: &mut usize,
    size_width: usize,
    index: &mut u64,
    thresh: usize,
) -> u64 {
    // get next pointer from table, making sure we don't grab a
    // pointer pointing into overlapping region.
    // Also increment (start) index in corresponding text
    let mut location = next_pointer(table, offset, size_width);
    if location == u64::MAX {
        return location;
    }
    *index += 1;
    while location >= thresh as u64 {
        location = next_pointer(table, offset, size_width);
        if location == u64::MAX {
            return location;
        }
        *index += 1;
    }
    return location;
}

pub fn in_memory_position(text: &[u8], table: &[u8], query: &[u8], size_width: usize) -> usize {
    let (mut left, mut right) = (0, table.len() / size_width);
    // let size_width = table.len() / text.len();

    while left < right {
        let mid = (left + right) / 2;
        // mid indexes into text, so must mult by size_width to correctly index into table
        let ptr = get_pointer(table, mid * size_width, size_width) as usize;
        if ptr > text.len() {
            println!(
                "ptr: {}, mid: {}, tl: {}, l: {}, r: {}, ratio: {}",
                ptr,
                mid,
                text.len(),
                left,
                right,
                size_width
            )
        }
        if query < &text[ptr..] {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    left
}

// pub fn from_bytes(input: Vec<u8>, size_width: usize) -> Vec<u64> {
//     // println!("S {}", input.len());
//     assert!(input.len() % size_width == 0);
//     let mut bytes: Vec<u64> = Vec::with_capacity(input.len() / size_width);

//     let mut tmp = [0u8; 8];
//     // todo learn rust macros, hope they're half as good as lisp marcos
//     // and if they are then come back and optimize this
//     for i in 0..input.len() / size_width {
//         tmp[..size_width].copy_from_slice(&input[i * size_width..i * size_width + size_width]);
//         bytes.push(u64::from_le_bytes(tmp));
//     }

//     bytes
// }

pub fn estimate_ratio(bytes: &Vec<u8>) -> usize {
    // expect the first value represented in bytes to be < 256
    // so the ratio is the first non-zero byte after the initial byte.
    // this will fail if the second value represented has least sig. byte 0
    // let tail = &bytes[1..];
    if let Some((ratio, _)) = bytes
        .iter()
        .enumerate()
        .find(|(i, x)| (*i > 0) & (x > &&0u8))
    {
        ratio
    } else {
        0
    }
}
