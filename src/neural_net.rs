use candle_core::{Device, Tensor, Var};
use candle_nn::encoding;

const LETTERS: &[char] = &[
    '.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
];

// Run the neural network training.
pub fn run(data: &Vec<String>, device: &Device, iterations: usize) {
    let (input, target) = tokenize(data, device);

    // Randomized starting weights.
    let mut weights = Var::rand(0f32, 1f32, (27, 27), device).unwrap();

    println!(
        "running {} gradient descent training iterations",
        iterations
    );
    for count in 0..iterations {
        let loss = forward_pass(&input, &target, &weights);
        println!("run {} loss {}", count + 1, loss);

        weights.backward().unwrap().remove(&weights);
        let loss_grad = loss.backward().unwrap();
        let weights_grad = loss_grad.get(&weights).unwrap();

        weights = Var::from_tensor(
            &weights
                .broadcast_sub(
                    &weights_grad
                        .broadcast_mul(&Tensor::new(&[50f32], device).unwrap())
                        .unwrap(),
                )
                .unwrap(),
        )
        .unwrap();
    }
}

// A forward pass of the input over the weights and loss calculation.
fn forward_pass(input: &Tensor, target: &Tensor, weights: &Tensor) -> Tensor {
    // One-hot encoding of all inputs with a depth of 27 for the 26 letters + delimiter.
    let input_enc = encoding::one_hot(input.clone(), 27, 1f32, 0f32).unwrap();

    let logits = input_enc.matmul(&weights).unwrap();
    let counts = logits.exp().unwrap();
    let sum = counts.clone().sum_keepdim(1).unwrap();
    let probs = counts.clone().broadcast_div(&sum.clone()).unwrap();

    // Calculate negative log-likelihood loss.
    // Use a one-hot of the targets to select their probabilities, p, then calculate -log(p).mean().
    let target_onehot = encoding::one_hot(target.clone(), 27, 1f32, 0f32).unwrap();
    let selected_probs = probs.broadcast_mul(&target_onehot).unwrap().sum(1).unwrap();
    let loss = selected_probs
        .log()
        .unwrap()
        .neg()
        .unwrap()
        .mean_all()
        .unwrap();

    return loss;
}

// Tokenize a list of strings for neural network training.
fn tokenize(words: &Vec<String>, device: &Device) -> (Tensor, Tensor) {
    let delimiter: char = '.';
    let mut input: Vec<u8> = vec![];
    let mut target: Vec<u8> = vec![];

    for word in words {
        let mut index = 0;
        let chars: Vec<char> = word.chars().collect();
        while index < word.len() {
            if index == 0 {
                input.push(ltoi(delimiter.clone()));
                target.push(ltoi(chars[index]));
                index += 1;
                continue;
            }

            if index == word.len() - 1 {
                input.push(ltoi(chars[index]));
                target.push(ltoi(delimiter.clone()));
                break;
            }

            input.push(ltoi(chars[index]));
            target.push(ltoi(chars[index + 1]));
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

// Convert a letter into an integer for data normalization.
// . -> 0, a -> 1, b -> 2, ...
// NOTE: Everything that's not a letter is compressed onto the letter 'z'.
fn ltoi(letter: char) -> u8 {
    return LETTERS.iter().position(|&c| c == letter).unwrap_or(26) as u8;
}

// Convert an integer to a letter.
fn itol(index: u8) -> char {
    return LETTERS.get(index as usize).unwrap_or(&'z').clone();
}
