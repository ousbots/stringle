use crate::{
    app::{
        device,
        message::{LossType, ModelMessage},
        options::Options,
    },
    error::VibeError,
    models::{data, model::Model},
};

use candle_core::{Device, Tensor, Var};
use candle_nn::encoding;
use rand::Rng;
use std::sync::mpsc::Sender;

pub struct NN {
    device: Device,
    data: Vec<String>,
    weights: Var,
}

// Run the neural network training.
impl Model for NN {
    fn train(&mut self, options: &Options, sender: Sender<ModelMessage>) -> Result<(), VibeError> {
        let (input, target) = tokenize(&self.data, &self.device)?;

        // The training rounds.
        //
        // Rounds are forward pass, backpropagate, then adjust the weights based on the calculated loss.
        for count in 0..options.iterations {
            let loss = self.forward_pass(&input, &target)?;

            self.backward_pass(&loss, options)?;

            // Send progress updates through the channel
            let _ = sender.send(ModelMessage::Progress {
                loss_type: LossType::Training,
                iteration: count,
                loss: loss.to_vec0::<f32>()?,
            });
        }

        let _ = sender.send(ModelMessage::Finished);

        Ok(())
    }

    // Generate new words from the trained weights.
    //
    // Sample from the trained weights to generate new similar strings. Sampling is running a loop
    // that: encodes the input position to a one-hot, use it to mask the trained weights, calculate
    // probabilities from the extracted weights, then sample the calculated probabilities. The
    // result is used as the input on the next round until a position of 0 (the delimiter) is
    // reached, which is the end of the word.
    fn generate(&mut self, options: &Options, sender: Sender<ModelMessage>) -> Result<(), VibeError> {
        for _ in 0..options.generate {
            let mut position: u8 = 0;
            let mut output: String = "".to_string();

            loop {
                let position_enc = encoding::one_hot(Tensor::new(&[position], &self.device)?, 27, 1f32, 0f32)?;
                let logits = position_enc.matmul(&self.weights)?;
                let counts = logits.exp()?;
                let probs = counts.broadcast_div(&counts.sum_keepdim(1)?)?;

                // Random sample from the probability.
                position = random_sample(&probs)?;
                if position == 0 {
                    break;
                }
                output.push(data::itol(position));
            }

            let _ = sender.send(ModelMessage::Generated {
                text: format!("    {}", output),
            });
        }

        let _ = sender.send(ModelMessage::Finished);

        Ok(())
    }
}

impl NN {
    pub fn init(options: &Options) -> Result<Self, VibeError> {
        let device = device::open_device(&options.device)?;

        Ok(Self {
            data: data::parse_data(&options.data)?,
            // Randomized starting weights that will be updated every training round.
            weights: Var::rand(0f32, 1f32, (27, 27), &device)?,
            device: device,
        })
    }

    // A forward pass of the input over the weights and loss calculation.
    //
    // The forward pass involves encoding the inputs as a one-hot vector, which is like a mask or
    // identity tensor of the input over the input space. The one-hot is cross multiplied with the
    // weights, extracting the weight values that correspond to the input. These weights are then
    // squared to ensure they're positive. These squared weight values are commonly called log-counts,
    // or logits. Finally, the loss value is calculated with a weights decay as:
    // (logits^2 / sum(logits)) - 0.1 * (w^2).mean().
    fn forward_pass(&mut self, input: &Tensor, target: &Tensor) -> Result<Tensor, VibeError> {
        // One-hot encoding of all inputs with a depth of 27 for the 26 letters + delimiter.
        let input_enc = encoding::one_hot(input.clone(), 27, 1f32, 0f32)?;

        let logits = input_enc.matmul(&self.weights)?;
        let counts = logits.exp()?;
        let probs = counts.broadcast_div(&counts.sum_keepdim(1)?)?;

        // Calculate negative log-likelihood loss.
        // Use a one-hot of the targets to select their probabilities, p, then calculate the loss with
        // a weight decay: -log(p).mean() + 0.01 * (w^2).mean().
        let target_onehot = encoding::one_hot(target.clone(), 27, 1f32, 0f32)?;
        let selected_probs = probs.broadcast_mul(&target_onehot)?.sum(1)?;
        let loss = selected_probs
            .log()?
            .neg()?
            .mean_all()?
            .add(&self.weights.powf(2.0)?.mean_all()?.affine(0.01, 0.0)?)?;

        Ok(loss)
    }

    fn backward_pass(&mut self, loss: &Tensor, options: &Options) -> Result<(), VibeError> {
        self.weights.backward()?.remove(&self.weights);
        let loss_grad = loss.backward()?;
        let weights_grad = loss_grad
            .get(self.weights.as_tensor())
            .ok_or_else(|| VibeError::new("missing loss gradient"))?;

        self.weights = Var::from_tensor(
            &self
                .weights
                .broadcast_sub(&weights_grad.broadcast_mul(&Tensor::new(&[options.learn_rate], &self.device)?)?)?,
        )?;

        Ok(())
    }
}

// Take a random sample from the given probability tensor.
//
// In order to take the probability distribution into account, a cumulative sum of the
// probabilities is computed and the first index with a summed probability greater than a randomly
// chosen value is selected.
fn random_sample(probs: &Tensor) -> Result<u8, VibeError> {
    let random_val: f32 = rand::rng().random_range(0.0..1.0);

    let cumulative_sum = probs.cumsum(1)?.squeeze(0)?.to_vec1()?;
    for (index, &sum) in cumulative_sum.iter().enumerate() {
        if random_val <= sum {
            return Ok(index as u8);
        }
    }

    Ok((cumulative_sum.len() - 1) as u8)
}

// Tokenize a list of strings for neural network training.
//
// Strings are tokenized characterwise, only the previous character is considered when generating
// the next character. Each word in the input is surrounded by the delimiter character and then
// split into two lists, each character of the word is paired with it's next, so that the input
// tensor is every character of a word aligned with the target tensor of every next character. The
// characters are normalized to integers for later numerical calculations.
fn tokenize(words: &Vec<String>, device: &Device) -> Result<(Tensor, Tensor), VibeError> {
    let delimiter: char = data::LETTERS[0];
    let mut input: Vec<u8> = vec![];
    let mut target: Vec<u8> = vec![];

    for word in words {
        let mut index = 0;
        let chars: Vec<char> = word.chars().collect();
        while index < word.len() {
            if index == 0 {
                input.push(data::ltoi(delimiter.clone()));
                target.push(data::ltoi(chars[index]));
                index += 1;
                continue;
            }

            if index == word.len() - 1 {
                input.push(data::ltoi(chars[index]));
                target.push(data::ltoi(delimiter.clone()));
                break;
            }

            input.push(data::ltoi(chars[index]));
            target.push(data::ltoi(chars[index + 1]));
            index += 1;
        }
    }

    let input_len = input.len();
    let input_tensor = Tensor::from_vec(input, input_len, device)?;

    let target_len = target.len();
    let target_tensor = Tensor::from_vec(target, target_len, device)?;

    Ok((input_tensor, target_tensor))
}
