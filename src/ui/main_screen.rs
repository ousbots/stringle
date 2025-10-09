use crate::app::options::Options;
use crate::ui::{colors::Palette, logo};

use ratatui::{
    layout::{Alignment, Constraint, Layout, Rect, Spacing},
    style::{Color, Style, Stylize},
    symbols::Marker,
    text::{Line, Span},
    widgets::{Axis, Block, BorderType, Chart, Dataset, GraphType, Padding, Paragraph},
    Frame,
};

// Draw the main screen showing the options and model training statistics.
pub fn draw(frame: &mut Frame, options: &Options) {
    let area = frame.area();

    frame
        .buffer_mut()
        .set_style(area, (Palette::FG_COLOR, Palette::BG_COLOR));

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
            Span::raw(options.data.clone()),
        ]),
        Line::from(vec![
            Span::styled("device=", Style::default().fg(Color::Blue).bold()),
            Span::raw(options.device.clone()),
        ]),
        Line::from(vec![
            Span::styled("method=", Style::default().fg(Color::Blue).bold()),
            Span::raw(options.method.clone()),
        ]),
        Line::from(vec![
            Span::styled("iterations=", Style::default().fg(Color::Blue).bold()),
            Span::raw(options.iterations.to_string()),
        ]),
        Line::from(vec![
            Span::styled("generate=", Style::default().fg(Color::Blue).bold()),
            Span::raw(options.generate.to_string()),
        ]),
    ];

    let parameters_lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled("batch_size=", Style::default().fg(Color::Blue).bold()),
            Span::raw(options.batch_size.to_string()),
        ]),
        Line::from(vec![
            Span::styled("block_size=", Style::default().fg(Color::Blue).bold()),
            Span::raw(options.block_size.to_string()),
        ]),
        Line::from(vec![
            Span::styled("embedding_size=", Style::default().fg(Color::Blue).bold()),
            Span::raw(options.embedding_size.to_string()),
        ]),
        Line::from(vec![
            Span::styled("hidden_size=", Style::default().fg(Color::Blue).bold()),
            Span::raw(options.hidden_size.to_string()),
        ]),
        Line::from(vec![
            Span::styled("learn_rate=", Style::default().fg(Color::Blue).bold()),
            Span::raw(options.learn_rate.to_string()),
        ]),
    ];

    let config_layout = Layout::vertical([Constraint::Length(8), Constraint::Fill(1)]).spacing(Spacing::Space(1));
    let [logo_area, lower_config_area] = config_area.layout(&config_layout);

    let options_layout = Layout::vertical(Constraint::from_lengths([
        options_lines.len() as u16 + 2,
        parameters_lines.len() as u16 + 2,
    ]));
    let [options_area, parameters_area] = lower_config_area.layout(&options_layout);

    logo::render(frame, logo_area);

    let options_block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Palette::BORDER_COLOR)
        .padding(Padding::horizontal(1))
        .title("Options");

    frame.render_widget(Paragraph::new(options_lines).block(options_block), options_area);

    let parameters_block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Palette::BORDER_COLOR)
        .padding(Padding::horizontal(1))
        .title("Model Hyperparameters");

    frame.render_widget(
        Paragraph::new(parameters_lines).block(parameters_block),
        parameters_area,
    );

    render_loss(frame, model_area);
}

// Render the loss chart.
fn render_loss(frame: &mut Frame, area: Rect) {
    let datasets = vec![
        Dataset::default()
            .name("Training Loss")
            .marker(Marker::Braille)
            .graph_type(GraphType::Scatter)
            .style(Palette::TRAINING_LOSS_COLOR)
            .data(&TRAINING_LOSS_DATA),
        Dataset::default()
            .name("Validation Loss")
            .marker(Marker::Dot)
            .graph_type(GraphType::Scatter)
            .style(Palette::VALIDATION_LOSS_COLOR)
            .data(&VALIDATION_LOSS_DATA),
    ];

    let chart = Chart::new(datasets)
        .style(Style::default().fg(Palette::FG_COLOR).bg(Palette::BG_COLOR))
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Palette::BORDER_COLOR)
                .style(Style::default().fg(Palette::FG_COLOR))
                .title(Line::from("Loss").cyan().bold().centered()),
        )
        .x_axis(
            Axis::default()
                .bounds([0., 300.])
                .style(Style::default().fg(Palette::FG_COLOR))
                .labels(["0", "100", "200", "300"]),
        )
        .y_axis(
            Axis::default()
                .bounds([0., 40.])
                .style(Style::default().fg(Palette::FG_COLOR))
                .labels(["0", "10", "20", "30", "40"]),
        );

    frame.render_widget(chart, area);
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
