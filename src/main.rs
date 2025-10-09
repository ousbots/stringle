mod app;
mod error;
mod models;
mod ui;

use app::app::App;

use error::VibeError;

fn main() -> Result<(), VibeError> {
    let mut app = App::new();
    app.run()?;

    let data = models::data::parse_data(&app.options.data)?;

    match app.options.method.as_str() {
        "nn" => models::neural_net::run(data, app.device, app.options)?,
        "mlp" => models::mlp::run(data, app.device, app.options)?,
        _ => println!("invalid method option: {}", app.options.method),
    }

    Ok(())
}
