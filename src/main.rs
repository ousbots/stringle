mod app;
mod data;
mod device;
mod errors;
mod mlp;
mod neural_net;
mod options;
mod ui;

use errors::VibeError;

fn main() -> Result<(), VibeError> {
    let mut app = app::App::new();

    options::parse_args(&mut app.options)?;

    app.run()?;

    device::open_device(&mut app)?;
    let data = data::parse_data(&app.options.data)?;

    match app.options.method.as_str() {
        "nn" => neural_net::run(data, app.device, app.options)?,
        "mlp" => mlp::run(data, app.device, app.options)?,
        _ => println!("invalid method option: {}", app.options.method),
    }

    Ok(())
}
