use std::env;
use std::process;

// User provided runtime arguments.
#[derive(Debug)]
pub struct Options {
    pub data: String,
    pub device: String,
    pub iterations: usize,
}

const DEFAULT_DATA_PATH: &str = "data/names.txt";
const DEFAULT_DEVICE: &str = "cpu";
const DEFAULT_ITERATIONS: usize = 100;

// Parse the command line options.
pub fn parse_args() -> Options {
    let mut options = Options {
        data: DEFAULT_DATA_PATH.to_string(),
        device: DEFAULT_DEVICE.to_string(),
        iterations: DEFAULT_ITERATIONS,
    };

    let mut args: Vec<String> = env::args().collect();
    args.reverse();
    args.pop();

    while let Some(arg) = args.pop() {
        match arg.as_str() {
            "--data" => {
                if let Some(path) = args.pop() {
                    options.data = path;
                } else {
                    println!("missing the path portion of the --data flag");
                    print_help();
                    process::exit(-1);
                }
            }
            "--device" => {
                if let Some(path) = args.pop() {
                    options.device = path;
                } else {
                    println!("missing the device portion of the --device flag");
                    print_help();
                    process::exit(-1);
                }
            }
            "--iterations" => {
                if let Some(iterations) = args.pop() {
                    options.iterations = str::parse::<usize>(iterations.as_str()).unwrap();
                } else {
                    println!("missing the iterations portion of the --iterations flag");
                    print_help();
                    process::exit(-1);
                }
            }
            _ => {
                print_help();
                process::exit(-1);
            }
        }
    }

    return options;
}

// Print a usage help message.
fn print_help() {
    println!("usage:");
    println!("command");
    println!("\t--data <data path> (default: {})", DEFAULT_DATA_PATH);
    println!(
        "\t--device <device> [metal, cuda, cpu] (default: {})",
        DEFAULT_DEVICE
    );
    println!(
        "\t--iterations <iterations> (default: {})",
        DEFAULT_ITERATIONS
    );
}
