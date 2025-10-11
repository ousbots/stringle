use crate::{
    app::{
        device,
        message::{LossType, ModelMessage},
        options::Options,
    },
    error::VibeError,
    models::{data, model::Model},
};

use candle_core::{backprop::GradStore, Device, Tensor, Var};
use candle_nn::{loss, ops};
use rand::{seq::SliceRandom, Rng};
use std::sync::mpsc::Sender;

// The vocabulary is hardcoded to the 26 letters plus the special delimiter character.
const VOCAB_SIZE: usize = 27;

pub struct MLP {
    device: Device,
    data: Vec<String>,
    c: Var,
    weights_1: Var,
    biases_1: Var,
    weights_2: Var,
    biases_2: Var,
}

impl Model for MLP {
    fn train(&mut self, options: &Options, sender: Sender<ModelMessage>) -> Result<(), VibeError> {
        // Randomize the input data, then break it into different data sets.
        //
        // The three different data sets will be the training set, the dev (validation) set, and the
        // test set. The training set is used for model training, the dev set is a set of valid words
        // the model hasn't been trained on that we can validate against. And finally, the test set is
        // another set of words the model wasn't trained on that can be used sparingly to test the
        // model without influencing training.
        self.data.shuffle(&mut rand::rng());

        let training_end = (self.data.len() as f64 * 0.8).round() as usize;
        let dev_end = (self.data.len() as f64 * 0.9).round() as usize;

        let (input, target) = tokenize(&self.data[..training_end].to_vec(), &self.device, &options)?;
        let (input_dev, target_dev) = tokenize(&self.data[training_end..dev_end].to_vec(), &self.device, &options)?;
        let (_intput_test, _target_test) = tokenize(&self.data[dev_end..].to_vec(), &self.device, &options)?;

        // Training rounds.
        //
        // NOTE: the data is randomly batched every training round and all weights adjusted based on
        // the batch loss. This speeds up training by not having to calculate the entire gradient every
        // round. In the tradeoff between calculating the exact gradient every round versus running
        // more rounds, running more rounds shows better results.
        for count in 0..options.iterations {
            let batch_indices = Tensor::rand(0f32, input.dims()[0] as f32, (options.batch_size,), &self.device)?
                .to_dtype(candle_core::DType::U32)?;
            let batch = input.index_select(&batch_indices.flatten_all()?, 0)?;

            let loss = self.forward_pass(&batch, &target.index_select(&batch_indices.flatten_all()?, 0)?)?;

            self.backward_pass(&loss, &options)?;

            // Send progress updates through the channel
            let _ = sender.send(ModelMessage::Progress {
                loss_type: LossType::Training,
                iteration: count,
                loss: loss.to_vec0::<f32>()?,
            });

            // Send validation progress every few iterations.
            if count % 100 == 0 {
                let validation_loss = self.forward_pass(&input_dev, &target_dev)?;
                let _ = sender.send(ModelMessage::Progress {
                    loss_type: LossType::Validation,
                    iteration: count,
                    loss: validation_loss.to_vec0::<f32>()?,
                });
            }
        }

        let _ = sender.send(ModelMessage::Finished);

        Ok(())
    }

    fn generate(&mut self, options: &Options, sender: Sender<ModelMessage>) -> Result<(), VibeError> {
        for _ in 0..options.generate {
            let mut output: String = "".to_string();
            let mut context: Vec<u8> = vec![0; options.block_size];

            loop {
                let embeddings = self
                    .c
                    .index_select(&Tensor::new(context.clone(), &self.device)?.flatten_all()?, 0)?;

                let h = embeddings
                    .reshape(((), self.weights_1.dims()[0]))?
                    .matmul(&self.weights_1)?
                    .broadcast_add(&self.biases_1)?
                    .tanh()?;

                let logits = h.matmul(&self.weights_2)?.broadcast_add(&self.biases_2)?;

                let probs = ops::softmax(&logits, 1)?;

                let position = random_sample(&probs)?;
                if position == 0 {
                    break;
                }
                output.push(data::itol(position as u8));

                context.remove(0);
                context.push(position as u8);
            }

            let _ = sender.send(ModelMessage::Generated {
                text: format!("    {}", output),
            });
        }

        let _ = sender.send(ModelMessage::Finished);

        Ok(())
    }
}

impl MLP {
    pub fn init(options: &Options) -> Result<Self, VibeError> {
        let device = device::open_device(&options.device)?;

        Ok(Self {
            data: data::parse_data(&options.data)?,
            c: Var::rand(0f32, 1f32, (VOCAB_SIZE, options.embedding_size), &device)?,
            // The gain (max value) is discussed in the "Delving Deep into Rectifier" paper by Kaiming He.
            // gain: (5/3) * sqrt(embedding_size * block_size).
            weights_1: Var::rand(
                0f32,
                (5.0 / 3.0) / (options.embedding_size as f32 * options.block_size as f32).sqrt(),
                (options.embedding_size * options.block_size, options.hidden_size),
                &device,
            )?,
            biases_1: Var::rand(0f32, 0.01f32, options.hidden_size, &device)?,
            weights_2: Var::rand(0f32, 0.01f32, (options.hidden_size, VOCAB_SIZE), &device)?,
            biases_2: Var::zeros(VOCAB_SIZE, candle_core::DType::F32, &device)?,
            device: device,
        })
    }

    fn forward_pass(&self, input: &Tensor, target: &Tensor) -> Result<Tensor, VibeError> {
        // Embed the input into vectors.
        let embeddings = self.c.index_select(&input.flatten_all()?, 0)?;

        // Hidden layer pre-activation with weights and biases and activation with tanh.
        let h = embeddings
            .reshape(((), self.weights_1.dims()[0]))?
            .matmul(&self.weights_1)?
            .broadcast_add(&self.biases_1)?
            .tanh()?;

        // Output layer.
        let logits = h.matmul(&self.weights_2)?.broadcast_add(&self.biases_2)?;

        // Loss function.
        Ok(loss::cross_entropy(
            &logits,
            &target.to_dtype(candle_core::DType::I64)?,
        )?)
    }

    fn backward_pass(&mut self, loss: &Tensor, options: &Options) -> Result<(), VibeError> {
        let loss_grad = loss.backward()?;

        backward_pass_parameter(&mut self.c, &loss_grad, options.learn_rate, &self.device)?;
        backward_pass_parameter(&mut self.weights_1, &loss_grad, options.learn_rate, &self.device)?;
        backward_pass_parameter(&mut self.biases_1, &loss_grad, options.learn_rate, &self.device)?;
        backward_pass_parameter(&mut self.weights_2, &loss_grad, options.learn_rate, &self.device)?;
        backward_pass_parameter(&mut self.biases_2, &loss_grad, options.learn_rate, &self.device)?;

        Ok(())
    }
}

// Run the gradient descent backward pass on a single parameter.
fn backward_pass_parameter(
    param: &mut Var,
    loss_grad: &GradStore,
    learn_rate: f32,
    device: &Device,
) -> Result<(), VibeError> {
    // Clear the gradient for this parameter.
    param.backward()?.remove(param.as_tensor());

    // Get the gradient from the loss gradient store.
    let gradient = loss_grad
        .get(param.as_tensor())
        .ok_or_else(|| VibeError::new("missing loss gradient"))?;

    // Compute the update: new_param = param - (gradient * learning_rate)
    let updated_param = param.broadcast_sub(&gradient.broadcast_mul(&Tensor::new(&[learn_rate], device)?)?)?;

    // Replace the parameter with the updated value.
    *param = Var::from_tensor(&updated_param)?;

    Ok(())
}

// Take a random sample from the given probability tensor.
//
// In order to take the probability distribution into account, a cumulative sum of the
// probabilities is computed and the first index with a summed probability greater than a randomly
// chosen value is selected.
fn random_sample(probs: &Tensor) -> Result<usize, VibeError> {
    let random_val: f32 = rand::rng().random_range(0.0..1.0);

    let cumulative_sum = probs.cumsum(1)?.squeeze(0)?.to_vec1()?;
    for (index, &sum) in cumulative_sum.iter().enumerate() {
        if random_val <= sum {
            return Ok(index);
        }
    }

    Ok(cumulative_sum.len() - 1)
}

// Tokenize a list of strings for neural network training.
//
// Strings are tokenized characterwise in blocks specified by options.block_size.
fn tokenize(words: &Vec<String>, device: &Device, options: &Options) -> Result<(Tensor, Tensor), VibeError> {
    let delimiter: char = data::LETTERS[0];
    let mut input: Vec<Vec<u8>> = vec![];
    let mut target: Vec<u8> = vec![];

    for word in words {
        let mut context: Vec<u8> = vec![0; options.block_size];

        let mut chars: Vec<char> = word.chars().collect();
        chars.push(delimiter);

        for letter in chars {
            let letter_value = data::ltoi(letter);
            input.push(context.clone());
            target.push(letter_value);

            context.remove(0);
            context.push(letter_value);
        }
    }

    let input_tensor = Tensor::from_vec(
        input.iter().flatten().copied().collect(),
        (input.len(), input[0].len()),
        device,
    )?;

    let target_len = target.len();
    let target_tensor = Tensor::from_vec(target, target_len, device)?;

    Ok((input_tensor, target_tensor))
}
