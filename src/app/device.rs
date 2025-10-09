use crate::app::app::App;
use crate::error::VibeError;

use candle_core::Device;

// Open the given device for data processing.
pub fn open_device(app: &mut App) -> Result<(), VibeError> {
    app.device = match app.options.device.trim().to_lowercase().as_str() {
        "cpu" => {
            println!("🧠 using the CPU for processing");
            Device::Cpu
        }
        "cuda" => {
            println!("🧠 using cuda for processing");
            Device::new_cuda(0).map_err(|e| VibeError::new(format!("unable to open cuda device: {}", e)))?
        }
        "metal" => {
            println!("🧠 using metal for processing");
            Device::new_metal(0).map_err(|e| VibeError::new(format!("unable to open metal device: {}", e)))?
        }
        _ => {
            println!(
                "🧠 invalid device name {}, falling back to CPU for processing",
                app.options.device
            );
            Device::Cpu
        }
    };

    Ok(())
}
