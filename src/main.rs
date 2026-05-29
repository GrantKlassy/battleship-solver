//! Battleship Solver — a probability-based solver with a live egui heatmap and a
//! parallel hybrid (exact enumeration + Monte-Carlo) engine.

mod app;
mod board;
mod solver;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1060.0, 780.0])
            .with_min_inner_size([760.0, 560.0])
            .with_title("Battleship Solver"),
        ..Default::default()
    };

    eframe::run_native(
        "Battleship Solver",
        native_options,
        Box::new(|cc| Box::new(app::App::new(cc))),
    )
}
