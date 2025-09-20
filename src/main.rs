mod data;
mod device;
mod neural_net;
mod options;

fn main() {
    let options = options::parse_args();
    let device = device::open_device(&options.device);
    let data = data::parse_data(&options.data);

    neural_net::run(&data, &device, &options);
}
