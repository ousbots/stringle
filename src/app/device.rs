use crate::app::app::App;
use crate::error::VibeError;

use candle_core::Device;

// Open the given device for data processing.
pub fn open_device(device: String) -> Result<Device, VibeError> {
    match device.trim().to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::new_cuda(0).map_err(|e| VibeError::new(format!("unable to open cuda device: {}", e)))?),
        "metal" => Ok(Device::new_metal(0).map_err(|e| VibeError::new(format!("unable to open metal device: {}", e)))?),
        _ => Err(VibeError::new("invalid device")),
    }
}
