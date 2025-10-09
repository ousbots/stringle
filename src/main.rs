mod app;
mod data;
mod device;
mod errors;
mod mlp;
mod neural_net;
mod options;

use errors::VibeError;

fn main() -> Result<(), VibeError> {
    let mut state = app::App::new();

    options::parse_args(&mut state.options)?;

    state.run()?;

    device::open_device(&mut state)?;
    let data = data::parse_data(&state.options.data)?;

    match state.options.method.as_str() {
        "nn" => neural_net::run(data, state.device, state.options)?,
        "mlp" => mlp::run(data, state.device, state.options)?,
        _ => println!("invalid method option: {}", state.options.method),
    }

    Ok(())
}
