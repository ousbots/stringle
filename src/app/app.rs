use crate::app::{
    message::{self, LossType, ModelMessage},
    options::{self, Options},
};
use crate::error::VibeError;
use crate::models;
use crate::ui::main_screen;

use crossterm::event::{self, KeyCode};
use ratatui::{
    backend::CrosstermBackend,
    crossterm::execute,
    crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    DefaultTerminal, Terminal,
};
use std::io;
use std::sync::mpsc::Receiver;
use std::thread::{self, JoinHandle};

pub struct App {
    pub terminal: DefaultTerminal,
    pub state: State,
    pub options: Options,
    pub loss_data: Vec<(f64, f64)>,
    pub validation_loss_data: Vec<(f64, f64)>,
    pub training_thread: Option<JoinHandle<Result<(), VibeError>>>,
    pub generated_data: Vec<String>,
}

#[derive(PartialEq)]
pub enum State {
    Main,
    Training,
    Generate,
    ShowGenerated,
    Exit,
}

impl App {
    pub fn new() -> Self {
        let backend = CrosstermBackend::new(io::stdout());
        let terminal = Terminal::new(backend).unwrap_or_else(|err| {
            panic!("unable to open terminal: {}", err);
        });

        Self {
            terminal: terminal,
            state: State::Main,
            options: Options::new(),
            loss_data: Vec::new(),
            validation_loss_data: Vec::new(),
            training_thread: None,
            generated_data: Vec::new(),
        }
    }

    pub fn draw_main(&mut self, show_generate: bool) -> Result<(), VibeError> {
        self.terminal.draw(|frame| {
            main_screen::draw(
                frame,
                &self.options,
                &self.loss_data,
                &self.validation_loss_data,
                &self.generated_data,
                show_generate,
            )
        })?;
        Ok(())
    }

    fn handle_input(&mut self) -> Result<(), VibeError> {
        let event = event::read()?;

        if event
            .as_key_press_event()
            .is_some_and(|key| key.code == KeyCode::Char('q') || key.code == KeyCode::Esc)
        {
            self.state = State::Exit;
        }

        if event
            .as_key_press_event()
            .is_some_and(|key| key.code == KeyCode::Char('t') || key.code == KeyCode::Enter)
        {
            if self.state != State::Training {
                self.state = State::Training;
            }
        }

        if event
            .as_key_press_event()
            .is_some_and(|key| key.code == KeyCode::Char('g'))
        {
            if self.state != State::Generate {
                self.state = State::Generate;
            }
        }

        if event
            .as_key_press_event()
            .is_some_and(|key| key.code == KeyCode::Char('p'))
        {
            if self.state == State::ShowGenerated {
                self.state = State::Main;
            } else {
                self.state = State::ShowGenerated;
            }
        }

        Ok(())
    }

    fn start_generation(&mut self) -> Result<(), VibeError> {
        Ok(())
    }

    fn start_training(&mut self) -> Result<(), VibeError> {
        // Only start training if not already running
        if self.training_thread.is_some() {
            return Ok(());
        }

        // Clear previous data
        self.loss_data.clear();
        self.validation_loss_data.clear();
        self.generated_data.clear();

        let data = models::data::parse_data(&self.options.data)?;
        let (sender, receiver) = message::create_channel();

        let method = self.options.method.clone();
        let options = self.options.clone();

        // Spawn the training thread.
        match method.as_str() {
            "nn" => {
                self.training_thread = Some(thread::spawn(move || models::neural_net::run(data, options, sender)));
            }
            "mlp" => {
                self.training_thread = Some(thread::spawn(move || models::mlp::run(data, options, sender)));
            }
            _ => return Err(VibeError::new(&format!("invalid method option: {}", method))),
        };

        self.process_training_messages(receiver)?;

        // Wait on the training thread.
        if let Some(thread) = self.training_thread.take() {
            _ = thread.join().map_err(|err| VibeError::new(format!("{:?}", err)))?;
            self.training_thread = None;
        }

        Ok(())
    }

    // Process all training messages, re-drawing as needed.
    fn process_training_messages(&mut self, receiver: Receiver<ModelMessage>) -> Result<(), VibeError> {
        while let Ok(message) = receiver.recv() {
            match message {
                ModelMessage::Progress {
                    loss_type,
                    iteration,
                    loss,
                } => match loss_type {
                    LossType::Training => {
                        self.loss_data.push((iteration as f64, loss as f64));
                        self.draw_main(false)?;
                    }
                    LossType::Validation => {
                        self.validation_loss_data.push((iteration as f64, loss as f64));
                        self.draw_main(false)?;
                    }
                },
                ModelMessage::Generated { value } => {
                    self.generated_data.push(value);
                }
                ModelMessage::Finished => {
                    break;
                }
            }
        }

        Ok(())
    }

    pub fn run(&mut self) -> Result<(), VibeError> {
        options::parse_args(&mut self.options)?;

        enable_raw_mode()?;
        execute!(self.terminal.backend_mut(), EnterAlternateScreen)?;

        loop {
            match self.state {
                State::Main => {
                    self.draw_main(false)?;
                }
                State::Training => {
                    self.start_training()?;
                    self.state = State::Main;
                }
                State::Generate => {
                    self.start_generation()?;
                    self.state = State::Main;
                }
                State::ShowGenerated => {
                    self.draw_main(true)?;
                }
                State::Exit => break,
            }

            self.handle_input()?;
        }

        disable_raw_mode()?;
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen)?;
        self.terminal.show_cursor()?;

        Ok(())
    }
}
