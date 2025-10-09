use crate::errors::VibeError;
use crate::options::Options;

use candle_core::Device;
use crossterm::event::{self, KeyCode};
use ratatui::{
    backend::CrosstermBackend,
    crossterm::event::{DisableMouseCapture, EnableMouseCapture},
    crossterm::execute,
    crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    layout::{Alignment, Constraint, Flex, Layout, Margin, Rect, Spacing},
    style::{Color, Style, Stylize},
    symbols::Marker,
    text::{Line, Span},
    widgets::{Axis, Block, BorderType, Chart, Dataset, GraphType, Padding, Paragraph},
    Frame, Terminal,
};
use std::io;
use std::iter::zip;

const LOGO_TOP: &str = "▄▀▀  █▀▀▄ ▀█▀ █  █ █▀▀▄ █▀▀▄";
const LOGO_BOT: &str = "▀▄▄▀ █▀▀   █  ▀▄▄▀ █▀▀▄ █▄▄▀";

const FG_COLOR: Color = Color::Rgb(246, 214, 187); // #F6D6BB
const BG_COLOR: Color = Color::Rgb(20, 20, 50); // #141432
const BORDER_COLOR: Color = Color::Rgb(255, 255, 160); // #FFFFA0

const TRAINING_LOSS_COLOR: Color = Color::Rgb(156, 227, 114);
const VALIDATION_LOSS_COLOR: Color = Color::Rgb(114, 214, 250);

pub struct App {
    pub state: State,
    pub options: Options,
    pub device: Device,
}

pub enum State {
    Home,
    Train,
    Generate,
    Exit,
}

impl App {
    pub fn new() -> Self {
        Self {
            state: State::Home,
            options: Options::new(),
            device: Device::Cpu,
        }
    }

    pub fn run(&mut self) -> Result<(), VibeError> {
        enable_raw_mode()?;
        execute!(io::stderr(), EnterAlternateScreen, EnableMouseCapture)?;

        let backend = CrosstermBackend::new(io::stderr());
        let mut terminal = Terminal::new(backend)?;

        loop {
            match self.state {
                State::Home => {
                    terminal.draw(|frame| draw_main(frame, self))?;
                }
                State::Train => {}
                State::Generate => {}
                State::Exit => break,
            }

            self.handle_input()?;

            if event::read()?
                .as_key_press_event()
                .is_some_and(|key| key.code == KeyCode::Char('q') || key.code == KeyCode::Esc)
            {
                break;
            }
        }

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
        terminal.show_cursor()?;

        Ok(())
    }

    fn handle_input(&mut self) -> Result<(), VibeError> {
        if event::read()?
            .as_key_press_event()
            .is_some_and(|key| key.code == KeyCode::Char('q') || key.code == KeyCode::Esc)
        {
            self.state = State::Exit;
        }

        Ok(())
    }
}

fn draw_main(frame: &mut Frame, app: &mut App) {
    let area = frame.area();

    frame.buffer_mut().set_style(area, (FG_COLOR, BG_COLOR));

    let main_layout = Layout::vertical([Constraint::Length(1), Constraint::Min(1)]);
    let content_layout = Layout::horizontal([Constraint::Length(30), Constraint::Fill(1)]);

    let [title_area, main_area] = main_layout.areas(area);
    let [config_area, model_area] = content_layout.areas(main_area);

    let title = Block::new()
        .title_alignment(Alignment::Center)
        .title("Generative Pretrained Turd".bold());
    frame.render_widget(title, title_area);

    let options_lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled("data=", Style::default().fg(Color::Blue).bold()),
            Span::raw(app.options.data.clone()),
        ]),
        Line::from(vec![
            Span::styled("device=", Style::default().fg(Color::Blue).bold()),
            Span::raw(app.options.device.clone()),
        ]),
        Line::from(vec![
            Span::styled("method=", Style::default().fg(Color::Blue).bold()),
            Span::raw(app.options.method.clone()),
        ]),
        Line::from(vec![
            Span::styled("iterations=", Style::default().fg(Color::Blue).bold()),
            Span::raw(app.options.iterations.to_string()),
        ]),
        Line::from(vec![
            Span::styled("generate=", Style::default().fg(Color::Blue).bold()),
            Span::raw(app.options.generate.to_string()),
        ]),
    ];

    let parameters_lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled("batch_size=", Style::default().fg(Color::Blue).bold()),
            Span::raw(app.options.batch_size.to_string()),
        ]),
        Line::from(vec![
            Span::styled("block_size=", Style::default().fg(Color::Blue).bold()),
            Span::raw(app.options.block_size.to_string()),
        ]),
        Line::from(vec![
            Span::styled("embedding_size=", Style::default().fg(Color::Blue).bold()),
            Span::raw(app.options.embedding_size.to_string()),
        ]),
        Line::from(vec![
            Span::styled("hidden_size=", Style::default().fg(Color::Blue).bold()),
            Span::raw(app.options.hidden_size.to_string()),
        ]),
        Line::from(vec![
            Span::styled("learn_rate=", Style::default().fg(Color::Blue).bold()),
            Span::raw(app.options.learn_rate.to_string()),
        ]),
    ];

    let config_layout = Layout::vertical([Constraint::Length(8), Constraint::Fill(1)]).spacing(Spacing::Space(1));
    let [logo_area, lower_config_area] = config_area.layout(&config_layout);

    let options_layout = Layout::vertical(Constraint::from_lengths([
        options_lines.len() as u16 + 2,
        parameters_lines.len() as u16 + 2,
    ]));
    let [options_area, parameters_area] = lower_config_area.layout(&options_layout);

    render_logo(frame, logo_area);

    let options_block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(BORDER_COLOR)
        .padding(Padding::horizontal(1))
        .title("Options");

    frame.render_widget(Paragraph::new(options_lines).block(options_block), options_area);

    let parameters_block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(BORDER_COLOR)
        .padding(Padding::horizontal(1))
        .title("Model Hyperparameters");

    frame.render_widget(
        Paragraph::new(parameters_lines).block(parameters_block),
        parameters_area,
    );

    render_loss(frame, model_area);
}

fn render_loss(frame: &mut Frame, area: Rect) {
    let datasets = vec![
        Dataset::default()
            .name("Training Loss")
            .marker(Marker::Braille)
            .graph_type(GraphType::Scatter)
            .style(TRAINING_LOSS_COLOR)
            .data(&TRAINING_LOSS_DATA),
        Dataset::default()
            .name("Validation Loss")
            .marker(Marker::Dot)
            .graph_type(GraphType::Scatter)
            .style(VALIDATION_LOSS_COLOR)
            .data(&VALIDATION_LOSS_DATA),
    ];

    let chart = Chart::new(datasets)
        .style(Style::default().fg(FG_COLOR).bg(BG_COLOR))
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(BORDER_COLOR)
                .style(Style::default().fg(FG_COLOR))
                .title(Line::from("Loss").cyan().bold().centered()),
        )
        .x_axis(
            Axis::default()
                .bounds([0., 300.])
                .style(Style::default().fg(FG_COLOR))
                .labels(["0", "100", "200", "300"]),
        )
        .y_axis(
            Axis::default()
                .bounds([0., 40.])
                .style(Style::default().fg(FG_COLOR))
                .labels(["0", "10", "20", "30", "40"]),
        );

    frame.render_widget(chart, area);
}

fn render_logo(frame: &mut Frame, area: Rect) {
    let area = area.inner(Margin::new(1, 0));
    let layout = Layout::vertical(Constraint::from_lengths([6, 1, 1])).flex(Flex::End);
    let [shadow_area, logo_top_area, logo_bottom_area] = area.layout(&layout);

    // Divide the logo into letter sections, rendering a block for each letter with a color based on row index.
    let letter_layout = Layout::horizontal(Constraint::from_lengths([5, 5, 4, 5, 5, 4]));
    for (row_index, row) in shadow_area.rows().enumerate() {
        for (rainbow, letter_area) in zip(Rainbow::ROYGBIV, row.layout_vec(&letter_layout)) {
            let color = rainbow.gradient_color(row_index);
            frame.render_widget(Block::new().style(color), letter_area);
        }
        // Render the logo truncated.
        frame.render_widget(LOGO_TOP, row);
    }

    frame.render_widget(Block::new().style(Color::Rgb(246, 214, 187)), logo_top_area);
    frame.render_widget(Block::new().style(Color::Rgb(246, 214, 187)), logo_bottom_area);
    frame.render_widget(LOGO_TOP, logo_top_area);
    frame.render_widget(LOGO_BOT, logo_bottom_area);
}

enum Rainbow {
    Red,
    Orange,
    Yellow,
    Green,
    Blue,
    Indigo,
    Violet,
}

impl Rainbow {
    const RED_GRADIENT: [u8; 6] = [41, 43, 50, 68, 104, 156];
    const GREEN_GRADIENT: [u8; 6] = [24, 30, 41, 65, 105, 168];
    const BLUE_GRADIENT: [u8; 6] = [55, 57, 62, 78, 113, 166];
    const AMBIENT_GRADIENT: [u8; 6] = [17, 18, 20, 25, 40, 60];

    const ROYGBIV: [Self; 7] = [
        Self::Red,
        Self::Orange,
        Self::Yellow,
        Self::Green,
        Self::Blue,
        Self::Indigo,
        Self::Violet,
    ];

    fn gradient_color(&self, row: usize) -> Color {
        let ambient = Self::AMBIENT_GRADIENT[row];
        let red = Self::RED_GRADIENT[row];
        let green = Self::GREEN_GRADIENT[row];
        let blue = Self::BLUE_GRADIENT[row];
        let blue_sat = Self::AMBIENT_GRADIENT[row].saturating_mul(6 - row as u8);
        let (r, g, b) = match self {
            Self::Red => (red, ambient, blue_sat),
            Self::Orange => (red, green / 2, blue_sat),
            Self::Yellow => (red, green, blue_sat),
            Self::Green => (ambient, green, blue_sat),
            Self::Blue => (ambient, ambient, blue.max(blue_sat)),
            Self::Indigo => (blue, ambient, blue.max(blue_sat)),
            Self::Violet => (red, ambient, blue.max(blue_sat)),
        };
        Color::Rgb(r, g, b)
    }
}

const TRAINING_LOSS_DATA: [(f64, f64); 30] = [
    (10., 29.50),
    (20., 20.15),
    (30., 12.60),
    (40., 9.45),
    (50., 6.37),
    (60., 5.81),
    (70., 5.25),
    (80., 4.96),
    (90., 4.81),
    (100., 4.75),
    (110., 4.62),
    (120., 4.54),
    (130., 4.42),
    (140., 4.39),
    (150., 4.36),
    (160., 4.33),
    (170., 4.30),
    (180., 4.28),
    (190., 4.26),
    (200., 4.23),
    (210., 4.21),
    (220., 4.18),
    (230., 4.16),
    (240., 4.14),
    (250., 4.12),
    (260., 4.11),
    (270., 4.09),
    (280., 4.07),
    (290., 4.06),
    (300., 4.06),
];

const VALIDATION_LOSS_DATA: [(f64, f64); 15] = [
    (20., 22.57),
    (40., 9.22),
    (60., 5.63),
    (80., 4.98),
    (100., 4.72),
    (120., 4.63),
    (140., 4.44),
    (160., 4.38),
    (180., 4.21),
    (200., 4.24),
    (220., 4.16),
    (240., 4.15),
    (260., 4.12),
    (280., 4.09),
    (300., 4.08),
];
