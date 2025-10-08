mod data;
mod device;
mod mlp;
mod neural_net;
mod options;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = options::parse_args()?;
    let device = device::open_device(&options.device);
    let data = data::parse_data(&options.data);

    match options.method.as_str() {
        "nn" => neural_net::run(data, device, options)?,
        "mlp" => mlp::run(data, device, options)?,
        _ => return Err(format!("invalid method option: {}", options.method).into()),
    }

    Ok(())
}
