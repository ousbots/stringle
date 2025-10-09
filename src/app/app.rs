use crate::app::{
    device,
    options::{self, Options},
};
use crate::error::VibeError;
use crate::ui::{generate_screen, main_screen};

use candle_core::Device;
use crossterm::event::{self, KeyCode};
use ratatui::{
    backend::CrosstermBackend,
    crossterm::execute,
    crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    DefaultTerminal, Terminal,
};
use std::io;

pub struct App {
    pub terminal: DefaultTerminal,
    pub state: State,
    pub options: Options,
    pub device: Device,
}

pub enum State {
    Main,
    Train,
    Generate,
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
            device: Device::Cpu,
        }
    }

    pub fn draw_main(&mut self) -> Result<(), VibeError> {
        self.terminal.draw(|frame| main_screen::draw(frame, &self.options))?;
        Ok(())
    }

    pub fn draw_generate(&mut self) -> Result<(), VibeError> {
        self.terminal.draw(|frame| generate_screen::draw(frame))?;
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
            self.state = State::Train;
        }

        if event
            .as_key_press_event()
            .is_some_and(|key| key.code == KeyCode::Char('g') || key.code == KeyCode::Char('p'))
        {
            self.state = State::Generate;
        }

        Ok(())
    }

    pub fn run(&mut self) -> Result<(), VibeError> {
        options::parse_args(&mut self.options)?;
        device::open_device(self)?;

        enable_raw_mode()?;
        execute!(self.terminal.backend_mut(), EnterAlternateScreen)?;

        loop {
            match self.state {
                State::Main => {
                    self.draw_main()?;
                }
                State::Train => {
                    self.draw_main()?;
                }
                State::Generate => {
                    self.draw_generate()?;
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
