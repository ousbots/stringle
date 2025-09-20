use candle_core::Device;
use std::fs;

mod neural_net;
mod options;

fn main() {
    let options = options::parse_args();
    let device = open_device(&options.device);
    let data = parse_data(&options.data);

    neural_net::run(&data, &device, &options);
}

// Open the given device for data processing.
fn open_device(device_name: &String) -> Device {
    let mut device = Device::Cpu;
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

    return device;
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
