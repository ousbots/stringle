use crate::errors::VibeError;
use candle_core::Device;

// Open the given device for data processing.
pub fn open_device(device_name: &String) -> Result<Device, VibeError> {
    let device = match device_name.trim().to_lowercase().as_str() {
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
            println!("🧠 invalid device name {device_name:?}, falling back to CPU for processing");
            Device::Cpu
        }
    };

    Ok(device)
}
