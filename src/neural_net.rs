use candle_core::{Device, Tensor, Var};
use candle_nn::encoding;
use rand::Rng;
use std::io;
use std::io::Write;

// Run the neural network training and generation.
pub fn run(data: &Vec<String>, device: &Device, options: &crate::options::Options) {
    let (input, target) = tokenize(data, device);

    // Randomized starting weights that will be updated every training round.
    let mut weights = Var::rand(0f32, 1f32, (27, 27), device).unwrap();
    let mut loss = Tensor::new(&[100f32], device).unwrap();

    println!(
        "running gradient descent training with {} iterations and a learn rate of {}\n",
        options.iterations, options.learn_rate
    );

    // The training rounds. Run a forward pass, backpropagate, then adjust the weights based on the
    // calculated loss.
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
                "{:.1}%\t",
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
    println!("post training loss {}\n", loss.to_vec0::<f32>().unwrap());

    println!("Generating {} new strings:", options.generate);

    // Sample from the trained weights to generate new similar strings.
    for _ in 0..options.generate {
        let mut position: u8 = 0;
        let mut output: String = "".to_string();

        loop {
            let position_enc =
                encoding::one_hot(Tensor::new(&[position], device).unwrap(), 27, 1f32, 0f32)
                    .unwrap();
            let logits = position_enc.matmul(&weights).unwrap();
            let counts = logits.exp().unwrap();
            let sum = counts.clone().sum_keepdim(1).unwrap();
            let probs = counts.clone().broadcast_div(&sum.clone()).unwrap();

            // Random sample from the probability.
            position = random_sample(&probs) as u8;
            output.push(crate::data::itol(position));

            if position == 0 {
                break;
            }
        }

        println!("{}", output);
    }
}

// A forward pass of the input over the weights and loss calculation.
//
// The forward pass is encoding the inputs as one-hots (sort of like an identity tensor that
// corresponds to the input value in the weights) which are cross multiplied with the weights, which
// are squared to ensure positive values, called log-counts. The loss value is then calculated as
// logits^2 / sum(logits).
fn forward_pass(input: &Tensor, target: &Tensor, weights: &Tensor) -> Tensor {
    // One-hot encoding of all inputs with a depth of 27 for the 26 letters + delimiter.
    let input_enc = encoding::one_hot(input.clone(), 27, 1f32, 0f32).unwrap();

    let logits = input_enc.matmul(&weights).unwrap();
    let counts = logits.exp().unwrap();
    let sum = counts.clone().sum_keepdim(1).unwrap();
    let probs = counts.clone().broadcast_div(&sum.clone()).unwrap();

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
    let input_tensor = Tensor::from_vec(input, input_len, device).unwrap_or_else(|error| {
        panic!("unable to create tensor: {error:?}");
    });

    let target_len = target.len();
    let target_tensor = Tensor::from_vec(target, target_len, device).unwrap_or_else(|error| {
        panic!("unable to create tensor: {error:?}");
    });

    return (input_tensor, target_tensor);
}
