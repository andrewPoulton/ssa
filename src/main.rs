use clap::{Parser, Subcommand};
use std::fs;
use std::io::prelude::*;
use std::io::BufReader;
use suffix_arrays::multithread_tokenize_jsonl;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Tokenize {
        #[clap(short, long)]
        chunk: String,
        #[clap(short, long)]
        out_path: String,
        #[clap(short, long)]
        text_key: String,
        #[clap(short, long)]
        model: String,
        #[clap(short, long, default_value_t = 0)]
        delim: i32,
        #[clap(short, long, default_value_t = 32)]
        num_workers: usize,
        #[clap(short, long, default_value_t = true)]
        lowercase: bool,
        #[clap(short, long, default_value_t = true)]
        remove_punc: bool,
        #[clap(short, long, default_value_t = 0)]
        show_progress: usize,
    },

    ReadAt {
        #[clap(short, long)]
        file: String,
        #[clap(short, long)]
        offset: u64,
    },
}

fn read_at(file: String, offset: u64) -> () {
    let mut file = fs::File::open(file).unwrap();
    let mut reader = BufReader::new(file);
    let _ = reader.seek(std::io::SeekFrom::Start(offset));
    let mut buf = String::new();
    let _ = reader.read_line(&mut buf);
    println!("{buf}");
}

fn main() -> () {
    let args = Args::parse();

    match &args.command {
        Commands::Tokenize {
            chunk,
            out_path,
            text_key,
            model,
            delim,
            num_workers,
            lowercase,
            remove_punc,
            show_progress,
        } => {
            let _ = multithread_tokenize_jsonl(
                chunk.to_owned(),
                out_path.to_owned(),
                text_key.to_owned(),
                model.to_owned(),
                *num_workers,
                *delim,
                *lowercase,
                *remove_punc,
                *show_progress,
            );
        }

        Commands::ReadAt { file, offset } => {
            let _ = read_at(file.to_owned(), *offset);
        }
    }
}
