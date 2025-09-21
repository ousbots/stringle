use candle_core::Device;

// Open the given device for data processing.
pub fn open_device(device_name: &String) -> Device {
    let mut device = Device::Cpu;
    match device_name.trim().to_lowercase().as_str() {
        "cpu" => {
            println!("🧠 using the CPU for processing");
            device = Device::Cpu;
        }
        "cuda" => {
            println!("🧠 using cuda for processing");
            device = Device::new_cuda(0).unwrap_or_else(|error| {
                panic!("unable to open cuda device: {error:?}");
            });
        }
        "metal" => {
            println!("🧠 using metal for processing");
            device = Device::new_metal(0).unwrap_or_else(|error| {
                panic!("unable to open metal device: {error:?}");
            });
        }
        _ => {
            println!("🧠 invalid device name {device_name:?}, falling back to CPU for processing")
        }
    }

    return device;
}
