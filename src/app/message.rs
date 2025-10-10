use std::sync::mpsc::{self, Receiver, Sender};

#[derive(Debug, Clone)]
pub enum LossType {
    Training,
    Validation,
}

// Message types for communication between training thread and UI
#[derive(Debug, Clone)]
pub enum TrainingMessage {
    Progress {
        loss_type: LossType,
        iteration: usize,
        loss: f32,
    },
    Generated {
        value: String,
    },
    Finished,
}

// Create a new channel pair for training communication
pub fn create_channel() -> (Sender<TrainingMessage>, Receiver<TrainingMessage>) {
    mpsc::channel()
}
