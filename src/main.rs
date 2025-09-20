use std::fs;

mod device;
mod neural_net;
mod options;

fn main() {
    let options = options::parse_args();
    let device = device::open_device(&options.device);
    let data = parse_data(&options.data);

    neural_net::run(&data, &device, &options);
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
