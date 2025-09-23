mod data;
mod device;
mod mlp;
mod neural_net;
mod options;

fn main() {
    let options = options::parse_args();
    let device = device::open_device(&options.device);
    let data = data::parse_data(&options.data);

    match options.method.as_str() {
        "nn" => neural_net::run(data, device, options),
        "mlp" => mlp::run(data, device, options),
        _ => panic!("invalid option: {}", options.method),
    }
}
