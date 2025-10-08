use crate::errors::VibeError;
use candle_core::{Device, Tensor, Var};
use candle_nn::loss;
use candle_nn::ops;
use rand::seq::SliceRandom;
use rand::Rng;
use std::io;
use std::io::Write;

// The vocabulary is hardcoded to the 26 letters plus the special delimiter character.
const VOCAB_SIZE: usize = 27;

pub fn run(mut data: Vec<String>, device: Device, options: crate::options::Options) -> Result<(), VibeError> {
    println!("😎 multilayer perceptron network");

    // Randomize the input data, then break it into different data sets.
    //
    // The three different data sets will be the training set, the dev (validation) set, and the
    // test set. The training set is used for model training, the dev set is a set of valid words
    // the model hasn't been trained on that we can validate against. And finally, the test set is
    // another set of words the model wasn't trained on that can be used sparingly to test the
    // model without influencing training.
    data.shuffle(&mut rand::rng());

    let training_end = (data.len() as f64 * 0.8).round() as usize;
    let dev_end = (data.len() as f64 * 0.9).round() as usize;

    let (input, target) = tokenize(&data[..training_end].to_vec(), &device, &options)?;
    let (input_dev, target_dev) = tokenize(&data[training_end..dev_end].to_vec(), &device, &options)?;
    let (_intput_test, _target_test) = tokenize(&data[dev_end..].to_vec(), &device, &options)?;

    // Model parameters.
    let c = Var::rand(0f32, 1f32, (VOCAB_SIZE, options.embedding_size), &device)?;
    // The gain (max value) is discussed in the "Delving Deep into Rectifier" paper by Kaiming He.
    // gain = (5/3) * sqrt(embedding_size * block_size).
    let weights_1 = Var::rand(
        0f32,
        (5.0 / 3.0) / (options.embedding_size as f32 * options.block_size as f32).sqrt(),
        (options.embedding_size * options.block_size, options.hidden_size),
        &device,
    )?;
    let biases_1 = Var::rand(0f32, 0.01f32, options.hidden_size, &device)?;
    let weights_2 = Var::rand(0f32, 0.01f32, (options.hidden_size, VOCAB_SIZE), &device)?;
    let biases_2 = Var::zeros(VOCAB_SIZE, candle_core::DType::F32, &device)?;

    let mut parameters = vec![c, weights_1, biases_1, weights_2, biases_2];

    println!(
        "🤯 running training with {} parameters, {} iterations, and hyperparameters:",
        parameters.iter().map(|elem| elem.elem_count()).sum::<usize>(),
        options.iterations,
    );
    println!("\tembedding layers {}, hidden layer neurons {}, training batch size {}, tokenization block size {}, learning rate {}",
        options.embedding_size,
        options.hidden_size,
        options.batch_size,
        options.block_size,
        options.learn_rate,
    );
    println!("");

    // Training rounds.
    //
    // NOTE: the data is randomly batched every training round and all weights adjusted based on
    // the batch loss. This speeds up training by not having to calculate the entire gradient every
    // round. In the tradeoff between calculating the exact gradient every round versus running
    // more rounds, running more rounds shows better results.
    for count in 0..options.iterations {
        let batch_indices = Tensor::rand(0f32, input.dims()[0] as f32, (options.batch_size,), &device)?
            .to_dtype(candle_core::DType::U32)?;
        let batch = input.index_select(&batch_indices.flatten_all()?, 0)?;

        let loss = forward_pass(
            &batch,
            &target.index_select(&batch_indices.flatten_all()?, 0)?,
            &parameters,
            &options,
        )?;

        backward_pass(&loss, &mut parameters, &device, &options)?;

        // Print iteration updates.
        if count % 100 == 0 {
            print!("{:.0}%\t", 100. * (count as f64) / (options.iterations as f64));
        }
        print!(".");
        if count > 0 && (count % 100 == 99 || count == options.iterations - 1) {
            print!("\n");
        }
        io::stdout().flush()?;
    }

    let training_loss = forward_pass(&input, &target, &parameters, &options)?;
    let validation_loss = forward_pass(&input_dev, &target_dev, &parameters, &options)?;

    println!("\n🤗🤗🤗🤗\n");
    println!("🤔 training loss {}", training_loss.to_vec0::<f32>()?);
    println!("🤔 validation loss {}", validation_loss.to_vec0::<f32>()?);
    println!("🫣 generating {} new strings:\n", options.generate);

    let c = parameters[0].as_tensor();
    let weights_1 = parameters[1].as_tensor();
    let biases_1 = parameters[2].as_tensor();
    let weights_2 = parameters[3].as_tensor();
    let biases_2 = parameters[4].as_tensor();

    for _ in 0..options.generate {
        let mut output: String = "".to_string();
        let mut context: Vec<u8> = vec![0; options.block_size];

        loop {
            let embeddings = c.index_select(&Tensor::new(context.clone(), &device)?.flatten_all()?, 0)?;

            let h = embeddings
                .reshape(((), weights_1.dims()[0]))?
                .matmul(&weights_1)?
                .broadcast_add(&biases_1)?
                .tanh()?;

            let logits = h.matmul(&weights_2)?.broadcast_add(&biases_2)?;

            let probs = ops::softmax(&logits, 1)?;

            let position = random_sample(&probs)?;
            if position == 0 {
                break;
            }
            output.push(crate::data::itol(position as u8));

            context.remove(0);
            context.push(position as u8);
        }

        println!("    {}", output);
    }

    Ok(())
}

fn forward_pass(
    input: &Tensor,
    target: &Tensor,
    parameters: &Vec<Var>,
    _options: &crate::options::Options,
) -> Result<Tensor, VibeError> {
    let c = parameters[0].as_tensor();
    let weights_1 = parameters[1].as_tensor();
    let biases_1 = parameters[2].as_tensor();
    let weights_2 = parameters[3].as_tensor();
    let biases_2 = parameters[4].as_tensor();

    // Embed the input into vectors.
    let embeddings = c.index_select(&input.flatten_all()?, 0)?;

    // Hidden layer pre-activation with weights and biases and activation with tanh.
    let h = embeddings
        .reshape(((), weights_1.dims()[0]))?
        .matmul(&weights_1)?
        .broadcast_add(&biases_1)?
        .tanh()?;

    // Output layer.
    let logits = h.matmul(&weights_2)?.broadcast_add(&biases_2)?;

    // Loss function.
    Ok(loss::cross_entropy(
        &logits,
        &target.to_dtype(candle_core::DType::I64)?,
    )?)
}

fn backward_pass(
    loss: &Tensor,
    parameters: &mut Vec<Var>,
    device: &Device,
    options: &crate::options::Options,
) -> Result<(), VibeError> {
    let loss_grad = loss.backward()?;

    // Zero the gradients on each parameter, then adjust the parameter by learning rate * loss gradient.
    for index in 0..parameters.len() {
        let param = parameters[index].as_tensor();
        param.backward()?.remove(param);

        let weights_grad = loss_grad
            .get(param)
            .ok_or_else(|| VibeError::new("missing loss gradient"))?;

        parameters[index] = Var::from_tensor(
            &param.broadcast_sub(&weights_grad.broadcast_mul(&Tensor::new(&[options.learn_rate], device)?)?)?,
        )?;
    }

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
fn tokenize(
    words: &Vec<String>,
    device: &Device,
    options: &crate::options::Options,
) -> Result<(Tensor, Tensor), VibeError> {
    let delimiter: char = crate::data::LETTERS[0];
    let mut input: Vec<Vec<u8>> = vec![];
    let mut target: Vec<u8> = vec![];

    for word in words {
        let mut context: Vec<u8> = vec![0; options.block_size];

        let mut chars: Vec<char> = word.chars().collect();
        chars.push(delimiter);

        for letter in chars {
            let letter_value = crate::data::ltoi(letter);
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
