mod app;
mod error;
mod models;
mod ui;

use app::app::App;
use error::VibeError;

fn main() -> Result<(), VibeError> {
    let mut app = App::new();
    app.run()?;

    Ok(())
}
