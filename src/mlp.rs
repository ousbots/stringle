use candle_core::{Device, Tensor, Var};
use candle_nn::loss;
use candle_nn::ops;
use rand::seq::SliceRandom;
use rand::Rng;
use std::io;
use std::io::Write;

// The vocabulary is hardcoded to the 26 letters plus the special delimiter character.
const VOCAB_SIZE: usize = 27;

pub fn run(mut data: Vec<String>, device: Device, options: crate::options::Options) {
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

    let (input, target) = tokenize(&data[..training_end].to_vec(), &device, &options);
    let (input_dev, target_dev) =
        tokenize(&data[training_end..dev_end].to_vec(), &device, &options);
    let (intput_test, target_test) = tokenize(&data[dev_end..].to_vec(), &device, &options);

    // Model parameters.
    let c = Var::rand(0f32, 1f32, (VOCAB_SIZE, options.embedding_size), &device).unwrap();
    let weights_1 = Var::rand(
        0f32,
        1f32,
        (
            options.embedding_size * options.block_size,
            options.hidden_size,
        ),
        &device,
    )
    .unwrap();
    let biases_1 = Var::rand(0f32, 1f32, options.hidden_size, &device).unwrap();
    let weights_2 = Var::rand(0f32, 1f32, (options.hidden_size, VOCAB_SIZE), &device).unwrap();
    let biases_2 = Var::rand(0f32, 1f32, VOCAB_SIZE, &device).unwrap();

    let mut parameters = vec![c, weights_1, biases_1, weights_2, biases_2];

    println!(
        "🤯 running gradient descent training on {} parameters with {} iterations and a learning rate of {}\n",
        parameters.iter().map(|elem| elem.elem_count()).sum::<usize>(), options.iterations, options.learn_rate
    );

    // Training rounds.
    //
    // NOTE: the data is randomly batched every training round and all weights adjusted based on
    // the batch loss. This speeds up training by not having to calculate the entire gradient every
    // round. In the tradeoff between calculating the exact gradient every round versus running
    // more rounds, running more rounds shows better results.
    for count in 0..options.iterations {
        let batch_indices =
            Tensor::rand(0f32, input.dims()[0] as f32, (options.batch_size,), &device)
                .unwrap()
                .to_dtype(candle_core::DType::U32)
                .unwrap();
        let batch = input
            .index_select(&batch_indices.flatten_all().unwrap(), 0)
            .unwrap();

        let loss = forward_pass(
            &batch,
            &target
                .index_select(&batch_indices.flatten_all().unwrap(), 0)
                .unwrap(),
            &parameters,
            &options,
        );

        backward_pass(&loss, &mut parameters, &device, &options);

        // Print iteration updates.
        if count % 100 == 0 {
            print!(
                "{:.0}%\t",
                100. * (count as f64) / (options.iterations as f64)
            );
        }
        print!(".");
        if count > 0 && (count % 100 == 99 || count == options.iterations - 1) {
            print!("\n");
        }
        io::stdout().flush().unwrap();
    }

    let training_loss = forward_pass(&input, &target, &parameters, &options);
    let validation_loss = forward_pass(&input_dev, &target_dev, &parameters, &options);

    println!("\n🤗🤗🤗🤗\n");
    println!(
        "🤔 training loss {}",
        training_loss.to_vec0::<f32>().unwrap()
    );
    println!(
        "🤔 validation loss {}",
        validation_loss.to_vec0::<f32>().unwrap()
    );
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
            let embeddings = c
                .index_select(
                    &Tensor::new(context.clone(), &device)
                        .unwrap()
                        .flatten_all()
                        .unwrap(),
                    0,
                )
                .unwrap();

            let h = embeddings
                .reshape(((), weights_1.dims()[0]))
                .unwrap()
                .matmul(&weights_1)
                .unwrap()
                .broadcast_add(&biases_1)
                .unwrap()
                .tanh()
                .unwrap();

            let logits = h
                .matmul(&weights_2)
                .unwrap()
                .broadcast_add(&biases_2)
                .unwrap();

            let probs = ops::softmax(&logits, 1).unwrap();

            let position = random_sample(&probs);
            if position == 0 {
                break;
            }
            output.push(crate::data::itol(position as u8));

            context.remove(0);
            context.push(position as u8);
        }

        println!("    {}", output);
    }
}

fn forward_pass(
    input: &Tensor,
    target: &Tensor,
    parameters: &Vec<Var>,
    _options: &crate::options::Options,
) -> Tensor {
    let c = parameters[0].as_tensor();
    let weights_1 = parameters[1].as_tensor();
    let biases_1 = parameters[2].as_tensor();
    let weights_2 = parameters[3].as_tensor();
    let biases_2 = parameters[4].as_tensor();

    // Embed the input into vectors.
    let embeddings = c.index_select(&input.flatten_all().unwrap(), 0).unwrap();

    // Hidden layer pre-activation with weights and biases and activation with tanh.
    let h = embeddings
        .reshape(((), weights_1.dims()[0]))
        .unwrap()
        .matmul(&weights_1)
        .unwrap()
        .broadcast_add(&biases_1)
        .unwrap()
        .tanh()
        .unwrap();

    // Output layer.
    let logits = h
        .matmul(&weights_2)
        .unwrap()
        .broadcast_add(&biases_2)
        .unwrap();

    // Loss function.
    return loss::cross_entropy(&logits, &target.to_dtype(candle_core::DType::I64).unwrap())
        .unwrap();
}

fn backward_pass(
    loss: &Tensor,
    parameters: &mut Vec<Var>,
    device: &Device,
    options: &crate::options::Options,
) {
    let loss_grad = loss.backward().unwrap();

    for index in 0..parameters.len() {
        let param = parameters[index].as_tensor();
        param.backward().unwrap().remove(param).unwrap();

        let weights_grad = loss_grad.get(param).unwrap();

        parameters[index] = Var::from_tensor(
            &param
                .broadcast_sub(
                    &weights_grad
                        .broadcast_mul(&Tensor::new(&[options.learn_rate], device).unwrap())
                        .unwrap(),
                )
                .unwrap(),
        )
        .unwrap();
    }
}

// Take a random sample from the given probability tensor.
//
// In order to take the probability distribution into account, a cumulative sum of the
// probabilities is computed and the first index with a summed probability greater than a randomly
// chosen value is selected.
fn random_sample(probs: &Tensor) -> usize {
    let random_val: f32 = rand::rng().random_range(0.0..1.0);

    let cumulative_sum = probs
        .cumsum(1)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec1()
        .unwrap();
    for (index, &sum) in cumulative_sum.iter().enumerate() {
        if random_val <= sum {
            return index;
        }
    }

    return cumulative_sum.len() - 1;
}

// Tokenize a list of strings for neural network training.
//
// Strings are tokenized characterwise in blocks specified by options.block_size.
fn tokenize(
    words: &Vec<String>,
    device: &Device,
    options: &crate::options::Options,
) -> (Tensor, Tensor) {
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
    )
    .unwrap();

    let target_len = target.len();
    let target_tensor = Tensor::from_vec(target, target_len, device).unwrap();

    return (input_tensor, target_tensor);
}
