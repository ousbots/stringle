/// The training data should be a list of strings separated by newlines. The data will be
/// normalized to be lowercase ascii characters between a-z, any other input characters will b
/// collapsed onto 'z'.
use crate::error::VibeError;

use std::fs;

// The normalized set of letters used for training. The '.' character is a special character used
// to designate the start and end of words.
pub const LETTERS: &[char] = &[
    '.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
    'w', 'x', 'y', 'z',
];

// Convert an normalized integer to a letter.
pub fn itol(index: u8) -> char {
    return LETTERS.get(usize::from(index)).unwrap_or(&'z').clone();
}

// Convert a letter into an integer for data normalization.
// NOTE: Input should be lowercase a-z and everything else is compressed onto the letter 'z'.
pub fn ltoi(letter: char) -> u8 {
    return LETTERS.iter().position(|&c| c == letter).unwrap_or(LETTERS.len() - 1) as u8;
}

// Read the data into a list of strings using newlines as a separator.
pub fn parse_data(path: &String) -> Result<Vec<String>, VibeError> {
    let content = fs::read_to_string(path).map_err(|e| VibeError::new(format!("unable to open {}: {}", path, e)))?;

    let items: Vec<String> = content
        .lines()
        .map(|elem| String::from(elem).trim().to_lowercase())
        .collect();

    Ok(items)
}
