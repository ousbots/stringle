mod data;
mod device;
mod errors;
mod mlp;
mod neural_net;
mod options;

use errors::VibeError;

fn main() -> Result<(), VibeError> {
    let options = options::parse_args()?;
    let device = device::open_device(&options.device)?;
    let data = data::parse_data(&options.data)?;

    match options.method.as_str() {
        "nn" => neural_net::run(data, device, options)?,
        "mlp" => mlp::run(data, device, options)?,
        _ => return Err(VibeError::new(format!("invalid method option: {}", options.method))),
    }

    Ok(())
}
