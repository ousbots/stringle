use std::env;
use std::process;

// User provided runtime arguments.
#[derive(Debug)]
pub struct Options {
    pub data: String,
    pub device: String,
    pub iterations: usize,
    pub learn_rate: f32,
    pub generate: usize,
}

const DEFAULT_DATA_PATH: &str = "data/names.txt";
const DEFAULT_DEVICE: &str = "cpu";
const DEFAULT_ITERATIONS: usize = 100;
const DEFAULT_LEARN_RATE: f32 = 50.0;
const DEFAULT_GENERATE: usize = 20;

// Parse the command line options.
pub fn parse_args() -> Options {
    let mut options = Options {
        data: DEFAULT_DATA_PATH.to_string(),
        device: DEFAULT_DEVICE.to_string(),
        iterations: DEFAULT_ITERATIONS,
        learn_rate: DEFAULT_LEARN_RATE,
        generate: DEFAULT_GENERATE,
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
                    println!("missing the number portion of the --iterations flag");
                    print_help();
                    process::exit(-1);
                }
            }
            "--learn-rate" => {
                if let Some(rate) = args.pop() {
                    options.learn_rate = str::parse::<f32>(rate.as_str()).unwrap();
                } else {
                    println!("missing the rate portion of the --learn-rate flag");
                    print_help();
                    process::exit(-1);
                }
            }
            "--generate" => {
                if let Some(count) = args.pop() {
                    options.generate = str::parse::<usize>(count.as_str()).unwrap();
                } else {
                    println!("missing the number portion of the --generate flag");
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
    println!("\t--data       <data path>      ({})", DEFAULT_DATA_PATH);
    println!("\t--device     <metal|cuda|cpu> ({})", DEFAULT_DEVICE);
    println!("\t--iterations <num>            ({})", DEFAULT_ITERATIONS);
    println!("\t--learn-rate <rate>           ({})", DEFAULT_LEARN_RATE);
    println!("\t--generate   <num>            ({})", DEFAULT_GENERATE);
}
