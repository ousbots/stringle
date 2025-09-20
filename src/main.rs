use candle_core::Device;
use std::env;
use std::fs;
use std::process;

mod neural_net;

// User provided runtime arguments.
#[derive(Debug)]
struct Options {
    data: String,
    device: Option<String>,
    iterations: usize,
}

fn main() {
    let options = parse_args();
    let device = open_device(&options.device);
    let data = parse_data(&options.data);

    neural_net::run(&data, &device, options.iterations);
}

fn open_device(device_name: &Option<String>) -> Device {
    let mut device = Device::Cpu;
    if let Some(device_name) = device_name {
        match device_name.trim().to_lowercase().as_str() {
            "cpu" => {
                println!("using the CPU for processing");
                device = Device::Cpu;
            }
            "cuda" => {
                println!("using cuda for processing");
                device = Device::new_cuda(0).unwrap_or_else(|error| {
                    panic!("unable to open cuda device: {error:?}");
                });
            }
            "metal" => {
                println!("using metal for processing");
                device = Device::new_metal(0).unwrap_or_else(|error| {
                    panic!("unable to open metal device: {error:?}");
                });
            }
            _ => {
                println!("invalid device name {device_name:?}, falling back to CPU for processing")
            }
        }
    } else {
        println!("defaulting to the CPU for processing");
    }

    return device;
}

// Parse the command line options.
fn parse_args() -> Options {
    let mut options = Options {
        data: "data/names.txt".to_string(),
        device: None,
        iterations: 100,
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
                    options.device = Some(path);
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

// Read the data into a list of strings using newlines as a separator.
fn parse_data(path: &String) -> Vec<String> {
    let items: Vec<String> = fs::read_to_string(path)
        .unwrap_or_else(|error| {
            panic!("unable to open {path}: {error:?}");
        })
        .lines()
        .map(String::from)
        .collect();

    return items;
}

fn print_help() {
    println!("usage:");
    println!("command");
    println!("\t--data <data path> (default: data/names.txt)");
    println!("\t--device <device> [metal, cuda, cpu] (default: cpu)")
}
