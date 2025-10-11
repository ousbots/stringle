use crate::{
    app::{message::ModelMessage, options::Options},
    error::VibeError,
};

use std::sync::mpsc::Sender;

pub const MODEL_NAME_MLP: &str = "mlp";
pub const MODEL_NAME_NN: &str = "nn";

pub trait Model: Send {
    fn train(&mut self, options: &Options, sender: Sender<ModelMessage>) -> Result<(), VibeError>;
    fn generate(&mut self, options: &Options, sender: Sender<ModelMessage>) -> Result<(), VibeError>;
}
