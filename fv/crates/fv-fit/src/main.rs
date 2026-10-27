use clap::Parser;

fn main() {
    let args = fv_fit::cli::Args::parse();
    if let Err(e) = fv_fit::cli::run(args) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
