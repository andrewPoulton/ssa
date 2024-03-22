pub mod build_impls;
pub mod search_impls;
pub mod table_builder;

pub use build_impls::{build_in_memory_array_impl, in_memory_merge_impl};
pub use search_impls::{find_duplicates_impl, find_off_disk_file_impl, find_off_disk_impl};
pub use table_builder::SuffixTable;
