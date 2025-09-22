use candle_core::{Device, Tensor, Var};
use candle_nn::encoding;
use rand::Rng;
use std::io;
use std::io::Write;

// Run the neural network training and generation.
pub fn run(data: &Vec<String>, device: &Device, options: &crate::options::Options) {
    println!("🥸 basic neural network");

    let (input, target) = tokenize(data, device);

    // Randomized starting weights that will be updated every training round.
    let mut weights = Var::rand(0f32, 1f32, (27, 27), device).unwrap();
    let mut loss = Tensor::new(&[100f32], device).unwrap();

    println!(
        "🤯 running gradient descent training with {} iterations and a learning rate of {}\n",
        options.iterations, options.learn_rate
    );

    // The training rounds.
    //
    // Rounds are forward pass, backpropagate, then adjust the weights based on the calculated loss.
    for count in 0..options.iterations {
        loss = forward_pass(&input, &target, &weights);

        weights.backward().unwrap().remove(&weights);
        let loss_grad = loss.backward().unwrap();
        let weights_grad = loss_grad.get(&weights).unwrap();

        weights = Var::from_tensor(
            &weights
                .broadcast_sub(
                    &weights_grad
                        .broadcast_mul(&Tensor::new(&[options.learn_rate], device).unwrap())
                        .unwrap(),
                )
                .unwrap(),
        )
        .unwrap();

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

    println!("\n🤗🤗🤗🤗\n");
    println!("🤔 post training loss {}", loss.to_vec0::<f32>().unwrap());
    println!("🫣 generating {} new strings:\n", options.generate);

    // Generate new words from the trained weights.
    //
    // Sample from the trained weights to generate new similar strings. Sampling is running a loop
    // that: encodes the input position to a one-hot, use it to mask the trained weights, calculate
    // probabilities from the extracted weights, then sample the calculated probabilities. The
    // result is used as the input on the next round until a position of 0 (the delimiter) is
    // reached, which is the end of the word.
    for _ in 0..options.generate {
        let mut position: u8 = 0;
        let mut output: String = "".to_string();

        loop {
            let position_enc =
                encoding::one_hot(Tensor::new(&[position], device).unwrap(), 27, 1f32, 0f32)
                    .unwrap();
            let logits = position_enc.matmul(&weights).unwrap();
            let counts = logits.exp().unwrap();
            let probs = counts
                .broadcast_div(&counts.sum_keepdim(1).unwrap())
                .unwrap();

            // Random sample from the probability.
            position = random_sample(&probs) as u8;
            if position == 0 {
                break;
            }
            output.push(crate::data::itol(position));
        }

        println!("    {}", output);
    }
}

// A forward pass of the input over the weights and loss calculation.
//
// The forward pass involves encoding the inputs as a one-hot vector, which is like a mask or
// identity tensor of the input over the input space. The one-hot is cross multiplied with the
// weights, extracting the weight values that correspond to the input. These weights are then
// squared to ensure they're positive. These squared weight values are commonly called log-counts,
// or logits. Finally, the loss value is calculated with a weights decay as:
// (logits^2 / sum(logits)) - 0.1 * (w^2).mean().
fn forward_pass(input: &Tensor, target: &Tensor, weights: &Tensor) -> Tensor {
    // One-hot encoding of all inputs with a depth of 27 for the 26 letters + delimiter.
    let input_enc = encoding::one_hot(input.clone(), 27, 1f32, 0f32).unwrap();

    let logits = input_enc.matmul(&weights).unwrap();
    let counts = logits.exp().unwrap();
    let probs = counts
        .broadcast_div(&counts.sum_keepdim(1).unwrap())
        .unwrap();

    // Calculate negative log-likelihood loss.
    // Use a one-hot of the targets to select their probabilities, p, then calculate the loss with
    // a weight decay: -log(p).mean() + 0.01 * (w^2).mean().
    let target_onehot = encoding::one_hot(target.clone(), 27, 1f32, 0f32).unwrap();
    let selected_probs = probs.broadcast_mul(&target_onehot).unwrap().sum(1).unwrap();
    let loss = selected_probs
        .log()
        .unwrap()
        .neg()
        .unwrap()
        .mean_all()
        .unwrap()
        .add(
            &weights
                .powf(2.0)
                .unwrap()
                .mean_all()
                .unwrap()
                .affine(0.01, 0.0)
                .unwrap(),
        )
        .unwrap();

    return loss;
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
// Strings are tokenized characterwise, only the previous character is considered when generating
// the next character. Each word in the input is surrounded by the delimiter character and then
// split into two lists, each character of the word is paired with it's next, so that the input
// tensor is every character of a word aligned with the target tensor of every next character. The
// characters are normalized to integers for later numerical calculations.
fn tokenize(words: &Vec<String>, device: &Device) -> (Tensor, Tensor) {
    let delimiter: char = crate::data::LETTERS[0];
    let mut input: Vec<u8> = vec![];
    let mut target: Vec<u8> = vec![];

    for word in words {
        let mut index = 0;
        let chars: Vec<char> = word.chars().collect();
        while index < word.len() {
            if index == 0 {
                input.push(crate::data::ltoi(delimiter.clone()));
                target.push(crate::data::ltoi(chars[index]));
                index += 1;
                continue;
            }

            if index == word.len() - 1 {
                input.push(crate::data::ltoi(chars[index]));
                target.push(crate::data::ltoi(delimiter.clone()));
                break;
            }

            input.push(crate::data::ltoi(chars[index]));
            target.push(crate::data::ltoi(chars[index + 1]));
            index += 1;
        }
    }

    let input_len = input.len();
    let input_tensor = Tensor::from_vec(input, input_len, device).unwrap();

    let target_len = target.len();
    let target_tensor = Tensor::from_vec(target, target_len, device).unwrap();

    return (input_tensor, target_tensor);
}
