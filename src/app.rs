//! The egui front-end: ship placement, the live probability heatmap, known-board
//! auto-play, and a manual "assistant" mode for solving a real external game.

use crate::board::*;
use crate::solver::*;
use eframe::egui;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::time::Duration;

#[derive(PartialEq, Eq)]
enum Mode {
    /// Manually placing your fleet before play (known-board mode only).
    Setup,
    /// Firing at the board / marking results.
    Play,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Orient {
    Horizontal,
    Vertical,
}

pub struct App {
    cfg: GameConfig,

    // Editable config (committed via "Apply & New Game").
    ui_w: i32,
    ui_h: i32,
    ui_ships: String,
    ui_gap: bool,
    cfg_error: Option<String>,

    mode: Mode,
    /// true: solver plays against a concrete board (placed or random) and shots
    /// auto-resolve. false: assistant — you mark hit/miss/sunk from a real game.
    known: bool,

    true_board: Option<TrueBoard>,
    states: Vec<CellState>,
    remaining: Vec<usize>,

    // Manual-placement scratch state (Setup).
    setup_occ: Vec<bool>,
    setup_cell_ship: Vec<Option<usize>>,
    setup_ships: Vec<Ship>,
    place_idx: usize,
    orient: Orient,

    // Assistant tool.
    sink_mode: bool,

    solve: Option<SolveResult>,
    dirty: bool,
    shots: u32,

    auto: bool,
    last_step_time: f64,
    speed_hz: f32,

    status: String,
    seed_counter: u64,
}

impl App {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let cfg = GameConfig::default();
        let mut app = Self {
            ui_w: cfg.width as i32,
            ui_h: cfg.height as i32,
            ui_ships: cfg
                .ships
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(","),
            ui_gap: cfg.gap_rule,
            cfg,
            cfg_error: None,
            mode: Mode::Setup,
            known: true,
            true_board: None,
            states: Vec::new(),
            remaining: Vec::new(),
            setup_occ: Vec::new(),
            setup_cell_ship: Vec::new(),
            setup_ships: Vec::new(),
            place_idx: 0,
            orient: Orient::Horizontal,
            sink_mode: false,
            solve: None,
            dirty: false,
            shots: 0,
            auto: false,
            last_step_time: 0.0,
            speed_hz: 6.0,
            status: String::new(),
            seed_counter: 0x1234_5678,
        };
        app.reset_game();
        app
    }

    fn next_seed(&mut self) -> u64 {
        // Simple LCG step — deterministic, no wall-clock needed.
        self.seed_counter = self
            .seed_counter
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.seed_counter
    }

    fn reset_game(&mut self) {
        let n = self.cfg.cells();
        self.states = vec![CellState::Unknown; n];
        self.remaining = self.cfg.ships.clone();
        self.solve = None;
        self.shots = 0;
        self.auto = false;
        self.sink_mode = false;
        self.dirty = true;

        if self.known {
            self.mode = Mode::Setup;
            self.setup_occ = vec![false; n];
            self.setup_cell_ship = vec![None; n];
            self.setup_ships = Vec::new();
            self.place_idx = 0;
            self.true_board = None;
            self.status = "Setup: click to place each ship (right-click or R to rotate), \
                           or hit Randomize."
                .into();
        } else {
            self.mode = Mode::Play;
            self.true_board = None;
            self.status =
                "Assistant mode: fire at the recommended cell, then mark the real result.".into();
        }
    }

    /// Build a `GameConfig` from the editable fields and start a fresh game.
    fn apply_config(&mut self) {
        let ships: Result<Vec<usize>, _> = self
            .ui_ships
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<usize>())
            .collect();

        let ships = match ships {
            Ok(v) => v,
            Err(_) => {
                self.cfg_error = Some("Ship list must be comma-separated numbers.".into());
                return;
            }
        };
        if ships.is_empty() || ships.iter().any(|&l| l == 0) {
            self.cfg_error = Some("Need at least one ship, all lengths ≥ 1.".into());
            return;
        }
        let (w, h) = (self.ui_w as usize, self.ui_h as usize);
        if ships.iter().any(|&l| l > w.max(h)) {
            self.cfg_error = Some("A ship is longer than the board's largest side.".into());
            return;
        }
        if ships.iter().sum::<usize>() > w * h {
            self.cfg_error = Some("Ships can't fit: total length exceeds the board.".into());
            return;
        }

        self.cfg_error = None;
        self.cfg = GameConfig {
            width: w,
            height: h,
            ships,
            gap_rule: self.ui_gap,
        };
        self.reset_game();
    }

    fn rotate(&mut self) {
        self.orient = match self.orient {
            Orient::Horizontal => Orient::Vertical,
            Orient::Vertical => Orient::Horizontal,
        };
    }

    /// Cells a ship of the current length would occupy if anchored at `anchor`.
    fn ship_cells_at(&self, anchor: usize, len: usize) -> Option<Vec<usize>> {
        let (x, y) = self.cfg.xy(anchor);
        let mut cells = Vec::with_capacity(len);
        for i in 0..len {
            let (cx, cy) = match self.orient {
                Orient::Horizontal => (x + i, y),
                Orient::Vertical => (x, y + i),
            };
            if cx >= self.cfg.width || cy >= self.cfg.height {
                return None;
            }
            cells.push(self.cfg.idx(cx, cy));
        }
        Some(cells)
    }

    fn try_place_setup(&mut self, anchor: usize) {
        if self.place_idx >= self.cfg.ships.len() {
            return;
        }
        let len = self.cfg.ships[self.place_idx];
        let Some(cells) = self.ship_cells_at(anchor, len) else {
            self.status = "Ship would run off the board there.".into();
            return;
        };
        if !can_place(&self.cfg, &self.setup_occ, &cells) {
            self.status = "Invalid spot — overlap or too close to another ship.".into();
            return;
        }
        let ship_idx = self.setup_ships.len();
        for &c in &cells {
            self.setup_occ[c] = true;
            self.setup_cell_ship[c] = Some(ship_idx);
        }
        self.setup_ships.push(Ship { len, cells });
        self.place_idx += 1;

        if self.place_idx == self.cfg.ships.len() {
            self.commit_setup();
        } else {
            let next = self.cfg.ships[self.place_idx];
            self.status = format!(
                "Placed. Next: ship {}/{} (length {}).",
                self.place_idx + 1,
                self.cfg.ships.len(),
                next
            );
        }
    }

    /// Randomly place any ships not yet placed in Setup, then commit.
    fn randomize_rest(&mut self) {
        let seed = self.next_seed();
        let mut rng = StdRng::seed_from_u64(seed);
        for i in self.place_idx..self.cfg.ships.len() {
            let len = self.cfg.ships[i];
            let candidates: Vec<Vec<usize>> = ship_placements(&self.cfg, len)
                .into_iter()
                .filter(|p| can_place(&self.cfg, &self.setup_occ, p))
                .collect();
            let Some(chosen) = candidates.choose(&mut rng) else {
                self.status = "Couldn't fit the rest — clear and try again.".into();
                return;
            };
            let ship_idx = self.setup_ships.len();
            for &c in chosen {
                self.setup_occ[c] = true;
                self.setup_cell_ship[c] = Some(ship_idx);
            }
            self.setup_ships.push(Ship {
                len,
                cells: chosen.clone(),
            });
        }
        self.place_idx = self.cfg.ships.len();
        self.commit_setup();
    }

    fn clear_setup(&mut self) {
        let n = self.cfg.cells();
        self.setup_occ = vec![false; n];
        self.setup_cell_ship = vec![None; n];
        self.setup_ships.clear();
        self.place_idx = 0;
        self.status = "Cleared. Place your ships.".into();
    }

    fn commit_setup(&mut self) {
        self.true_board = Some(TrueBoard {
            cell_ship: self.setup_cell_ship.clone(),
            ships: self.setup_ships.clone(),
        });
        self.start_play();
    }

    /// Generate a full random board and jump straight to play (known mode).
    fn randomize_board(&mut self) {
        let seed = self.next_seed();
        let mut rng = StdRng::seed_from_u64(seed);
        match TrueBoard::random(&self.cfg, &mut rng) {
            Some(tb) => {
                self.true_board = Some(tb);
                self.start_play();
            }
            None => {
                self.status = "Couldn't place the fleet randomly — adjust the configuration.".into();
            }
        }
    }

    fn start_play(&mut self) {
        let n = self.cfg.cells();
        self.states = vec![CellState::Unknown; n];
        self.remaining = self.cfg.ships.clone();
        self.mode = Mode::Play;
        self.shots = 0;
        self.solve = None;
        self.dirty = true;
        self.status = "Playing — fire at the brightest cell.".into();
    }

    /// Fire at `cell` against the known board and resolve the result.
    fn fire_known(&mut self, cell: usize) {
        if self.states[cell] != CellState::Unknown {
            return;
        }
        let Some(tb) = self.true_board.clone() else {
            return;
        };
        self.shots += 1;

        if let Some(si) = tb.cell_ship[cell] {
            self.states[cell] = CellState::Hit;
            let ship = &tb.ships[si];
            let sunk = ship
                .cells
                .iter()
                .all(|&c| matches!(self.states[c], CellState::Hit | CellState::Sunk));
            if sunk {
                for &c in &ship.cells {
                    self.states[c] = CellState::Sunk;
                }
                remove_one(&mut self.remaining, ship.len);
                self.status = format!("Hit — and sunk a length-{} ship!", ship.len);
            } else {
                self.status = "Hit!".into();
            }
        } else {
            self.states[cell] = CellState::Miss;
            self.status = "Miss.".into();
        }
        self.dirty = true;

        if self.remaining.is_empty() {
            self.status = format!("Solved! All ships sunk in {} shots.", self.shots);
            self.auto = false;
        }
    }

    /// Play the whole game out immediately (known mode), no animation.
    fn solve_instantly(&mut self) {
        let max_iters = self.cfg.cells() * 2;
        let mut guard = 0;
        while !self.remaining.is_empty() && guard < max_iters {
            let res = solve(&self.cfg, &self.states, &self.remaining);
            match res.best_cell {
                Some(bc) => self.fire_known(bc),
                None => break,
            }
            guard += 1;
        }
        self.dirty = true;
    }

    // ---- Assistant-mode marking (no known board) ----

    fn cycle_state(&mut self, cell: usize) {
        self.states[cell] = match self.states[cell] {
            CellState::Unknown => CellState::Miss,
            CellState::Miss => CellState::Hit,
            CellState::Hit => CellState::Unknown,
            // Sunk cells are only changed via the sink tool.
            CellState::Sunk => CellState::Sunk,
        };
        self.recount_assistant_shots();
        self.dirty = true;
    }

    fn mark_hit(&mut self, cell: usize) {
        if self.states[cell] != CellState::Sunk {
            self.states[cell] = CellState::Hit;
            self.recount_assistant_shots();
            self.dirty = true;
        }
    }

    /// Sink (or un-sink) the contiguous run of hits/sunks through `cell`.
    fn toggle_sink(&mut self, cell: usize) {
        match self.states[cell] {
            CellState::Hit => {
                let run = self.connected(cell, CellState::Hit);
                let len = run.len();
                for c in &run {
                    self.states[*c] = CellState::Sunk;
                }
                if remove_one(&mut self.remaining, len) {
                    self.status = format!("Marked a length-{} ship sunk.", len);
                } else {
                    self.status =
                        format!("Marked {} cells sunk (no length-{} ship in the fleet?).", len, len);
                }
            }
            CellState::Sunk => {
                let run = self.connected(cell, CellState::Sunk);
                let len = run.len();
                for c in &run {
                    self.states[*c] = CellState::Hit;
                }
                self.remaining.push(len);
                self.status = format!("Un-sunk a length-{} ship.", len);
            }
            _ => {}
        }
        self.recount_assistant_shots();
        self.dirty = true;
    }

    /// Orthogonally-connected run of cells sharing `target` state, from `start`.
    fn connected(&self, start: usize, target: CellState) -> Vec<usize> {
        let mut out = Vec::new();
        let mut seen = vec![false; self.cfg.cells()];
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(c) = stack.pop() {
            if self.states[c] != target {
                continue;
            }
            out.push(c);
            for nb in neighbors4(&self.cfg, c) {
                if !seen[nb] && self.states[nb] == target {
                    seen[nb] = true;
                    stack.push(nb);
                }
            }
        }
        out
    }

    fn recount_assistant_shots(&mut self) {
        self.shots = self
            .states
            .iter()
            .filter(|&&s| s != CellState::Unknown)
            .count() as u32;
    }

    // ---- Click dispatch ----

    fn on_primary(&mut self, idx: usize) {
        match self.mode {
            Mode::Setup => self.try_place_setup(idx),
            Mode::Play => {
                if self.known {
                    self.fire_known(idx);
                } else if self.sink_mode {
                    self.toggle_sink(idx);
                } else {
                    self.cycle_state(idx);
                }
            }
        }
    }

    fn on_secondary(&mut self, idx: usize) {
        match self.mode {
            Mode::Setup => self.rotate(),
            Mode::Play => {
                if !self.known && !self.sink_mode {
                    self.mark_hit(idx);
                }
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Recompute the heatmap whenever evidence changed (Play mode only).
        if self.dirty && self.mode == Mode::Play {
            self.solve = Some(solve(&self.cfg, &self.states, &self.remaining));
            self.dirty = false;
        }

        // Keyboard shortcuts.
        let (r_pressed, space_pressed) =
            ctx.input(|i| (i.key_pressed(egui::Key::R), i.key_pressed(egui::Key::Space)));
        if r_pressed && self.mode == Mode::Setup {
            self.rotate();
        }
        if space_pressed && self.mode == Mode::Play && self.known {
            if let Some(bc) = self.solve.as_ref().and_then(|s| s.best_cell) {
                self.fire_known(bc);
            }
        }

        self.side_panel(ctx);
        egui::CentralPanel::default().show(ctx, |ui| {
            self.legend(ui);
            ui.add_space(6.0);
            self.draw_grid(ui);
        });

        // Animated auto-play: fire one shot per tick.
        if self.auto && self.mode == Mode::Play && self.known {
            if self.remaining.is_empty() {
                self.auto = false;
            } else {
                let now = ctx.input(|i| i.time);
                let interval = 1.0 / self.speed_hz.max(0.5) as f64;
                if now - self.last_step_time >= interval {
                    self.last_step_time = now;
                    if let Some(bc) = self.solve.as_ref().and_then(|s| s.best_cell) {
                        self.fire_known(bc);
                    }
                }
                ctx.request_repaint_after(Duration::from_secs_f32(
                    (1.0 / self.speed_hz.max(0.5)).min(1.0),
                ));
            }
        }
    }
}

impl App {
    fn side_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("controls")
            .resizable(false)
            .exact_width(290.0)
            .show(ctx, |ui| {
                ui.heading("Battleship Solver");
                ui.add_space(4.0);

                egui::CollapsingHeader::new("Configuration")
                    .default_open(false)
                    .show(ui, |ui| {
                        egui::Grid::new("cfg_grid").num_columns(2).show(ui, |ui| {
                            ui.label("Width");
                            ui.add(egui::DragValue::new(&mut self.ui_w).clamp_range(5..=24));
                            ui.end_row();
                            ui.label("Height");
                            ui.add(egui::DragValue::new(&mut self.ui_h).clamp_range(5..=24));
                            ui.end_row();
                            ui.label("Ships");
                            ui.text_edit_singleline(&mut self.ui_ships);
                            ui.end_row();
                        });
                        ui.checkbox(&mut self.ui_gap, "No-touching rule (1-tile gap)");
                        if ui.button("Apply & New Game").clicked() {
                            self.apply_config();
                        }
                        if let Some(err) = &self.cfg_error {
                            ui.colored_label(egui::Color32::from_rgb(230, 80, 80), err);
                        }
                    });

                ui.separator();
                ui.heading("Game");

                let prev_known = self.known;
                ui.checkbox(&mut self.known, "Known board (solver plays a real board)");
                if self.known != prev_known {
                    self.reset_game();
                }
                if !self.known {
                    ui.label(
                        egui::RichText::new("Assistant mode: no hidden board — you mark results.")
                            .small()
                            .weak(),
                    );
                }

                ui.horizontal(|ui| {
                    if ui.button("New Game").clicked() {
                        self.reset_game();
                    }
                    if self.known && ui.button("Randomize Board").clicked() {
                        self.randomize_board();
                    }
                });

                ui.separator();
                self.mode_controls(ui);

                ui.separator();
                self.solver_panel(ui);

                ui.separator();
                ui.label(egui::RichText::new(&self.status).italics());
            });
    }

    fn mode_controls(&mut self, ui: &mut egui::Ui) {
        match self.mode {
            Mode::Setup => {
                ui.label(format!(
                    "Placing ship {}/{} (length {}).",
                    (self.place_idx + 1).min(self.cfg.ships.len()),
                    self.cfg.ships.len(),
                    self.cfg
                        .ships
                        .get(self.place_idx)
                        .copied()
                        .unwrap_or(0)
                ));
                ui.horizontal(|ui| {
                    let dir = match self.orient {
                        Orient::Horizontal => "Horizontal",
                        Orient::Vertical => "Vertical",
                    };
                    if ui.button(format!("Rotate (now {dir})")).clicked() {
                        self.rotate();
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Randomize Rest").clicked() {
                        self.randomize_rest();
                    }
                    if ui.button("Clear").clicked() {
                        self.clear_setup();
                    }
                });
                ui.label(
                    egui::RichText::new("Tip: right-click the board or press R to rotate.")
                        .small()
                        .weak(),
                );
            }
            Mode::Play => {
                ui.label(format!("Shots fired: {}", self.shots));
                ui.label(format!("Ships left: {}", fleet_str(&self.remaining)));

                if self.known {
                    ui.add_space(2.0);
                    ui.horizontal(|ui| {
                        let has_best = self
                            .solve
                            .as_ref()
                            .and_then(|s| s.best_cell)
                            .is_some();
                        if ui
                            .add_enabled(has_best, egui::Button::new("Fire Best (Space)"))
                            .clicked()
                        {
                            if let Some(bc) = self.solve.as_ref().and_then(|s| s.best_cell) {
                                self.fire_known(bc);
                            }
                        }
                        ui.checkbox(&mut self.auto, "Auto-play");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Speed");
                        ui.add(egui::Slider::new(&mut self.speed_hz, 1.0..=30.0).suffix(" /s"));
                    });
                    if ui.button("Solve Instantly").clicked() {
                        self.solve_instantly();
                    }
                } else {
                    ui.checkbox(&mut self.sink_mode, "Sink tool (click a hit run to sink it)");
                    ui.label(
                        egui::RichText::new(
                            "Left-click cycles Unknown → Miss → Hit. Right-click sets Hit.",
                        )
                        .small()
                        .weak(),
                    );
                }
            }
        }
    }

    fn solver_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Solver");
        if self.mode != Mode::Play {
            ui.label(
                egui::RichText::new("Heatmap appears once play begins.")
                    .small()
                    .weak(),
            );
            return;
        }
        match &self.solve {
            Some(s) if s.method != Method::Trivial => {
                ui.label(format!("Method: {}", s.method.label()));
                let count_label = match s.method {
                    Method::Exact => "configurations",
                    Method::MonteCarlo => "samples accepted",
                    Method::Trivial => "",
                };
                ui.label(format!("{}: {}", count_label, group_digits(s.configs)));
                if let Some(bc) = s.best_cell {
                    let (x, y) = self.cfg.xy(bc);
                    ui.label(format!(
                        "Best shot: {} ({}, {})",
                        coord_name(x, y),
                        x,
                        y
                    ));
                    ui.label(
                        egui::RichText::new(format!(
                            "Confidence: {:.1}%",
                            s.best_conf * 100.0
                        ))
                        .strong()
                        .color(egui::Color32::from_rgb(240, 200, 60)),
                    );
                }
            }
            _ => {
                if self.remaining.is_empty() {
                    ui.label("All ships sunk. 🎉");
                } else {
                    ui.label("No consistent configuration for the current marks.");
                }
            }
        }
    }

    fn legend(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let chip = |ui: &mut egui::Ui, color: egui::Color32, text: &str| {
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
                ui.painter().rect_filled(rect, egui::Rounding::same(3.0), color);
                ui.label(text);
                ui.add_space(8.0);
            };
            if self.mode == Mode::Setup {
                chip(ui, SHIP_COLOR, "Ship");
                chip(ui, WATER_COLOR, "Water");
            } else {
                chip(ui, MISS_COLOR, "Miss");
                chip(ui, HIT_COLOR, "Hit");
                chip(ui, SUNK_COLOR, "Sunk");
                chip(ui, heat(0.15), "low P");
                chip(ui, heat(0.9), "high P");
            }
        });
    }

    fn draw_grid(&mut self, ui: &mut egui::Ui) {
        let cfg = self.cfg.clone();
        let (cols, rows) = (cfg.width, cfg.height);

        let avail = ui.available_size();
        let cell = ((avail.x / cols as f32).min(avail.y / rows as f32))
            .floor()
            .clamp(16.0, 64.0);
        let size = egui::vec2(cell * cols as f32, cell * rows as f32);
        let (resp, painter) = ui.allocate_painter(size, egui::Sense::click());
        let origin = resp.rect.min;

        // Normalise the heatmap by its peak for contrast.
        let max_p = self
            .solve
            .as_ref()
            .map(|s| s.prob.iter().cloned().fold(0.0_f64, f64::max))
            .unwrap_or(0.0);
        let best = self.solve.as_ref().and_then(|s| s.best_cell);

        for y in 0..rows {
            for x in 0..cols {
                let idx = cfg.idx(x, y);
                let pos = origin + egui::vec2(x as f32 * cell, y as f32 * cell);
                let rect = egui::Rect::from_min_size(pos, egui::vec2(cell - 1.5, cell - 1.5));

                let (fill, text) = self.cell_visual(idx, max_p);
                painter.rect_filled(rect, egui::Rounding::same(2.0), fill);

                if Some(idx) == best {
                    painter.rect_stroke(
                        rect,
                        egui::Rounding::same(2.0),
                        egui::Stroke::new(2.5, egui::Color32::from_rgb(245, 210, 60)),
                    );
                }
                if let Some((label, color)) = text {
                    painter.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        label,
                        egui::FontId::proportional(cell * 0.34),
                        color,
                    );
                }
            }
        }

        // Resolve clicks to a cell, then mutate (after the immutable borrows end).
        let to_cell = |p: egui::Pos2| -> Option<usize> {
            let rel = p - origin;
            if rel.x < 0.0 || rel.y < 0.0 {
                return None;
            }
            let x = (rel.x / cell) as usize;
            let y = (rel.y / cell) as usize;
            if x < cols && y < rows {
                Some(cfg.idx(x, y))
            } else {
                None
            }
        };

        if resp.clicked() {
            if let Some(idx) = resp.interact_pointer_pos().and_then(to_cell) {
                self.on_primary(idx);
            }
        }
        if resp.secondary_clicked() {
            if let Some(idx) = resp
                .interact_pointer_pos()
                .or_else(|| resp.hover_pos())
                .and_then(to_cell)
            {
                self.on_secondary(idx);
            }
        }
    }

    /// Fill colour and optional centered label for a cell.
    fn cell_visual(&self, idx: usize, max_p: f64) -> (egui::Color32, Option<(String, egui::Color32)>) {
        if self.mode == Mode::Setup {
            return if self.setup_occ[idx] {
                (SHIP_COLOR, None)
            } else {
                (WATER_COLOR, None)
            };
        }
        match self.states[idx] {
            CellState::Miss => (MISS_COLOR, Some(("•".into(), egui::Color32::from_gray(180)))),
            CellState::Hit => (HIT_COLOR, Some(("✕".into(), egui::Color32::WHITE))),
            CellState::Sunk => (SUNK_COLOR, Some(("✕".into(), egui::Color32::from_gray(220)))),
            CellState::Unknown => {
                let p = self.solve.as_ref().map(|s| s.prob[idx]).unwrap_or(0.0);
                if max_p <= 0.0 {
                    return (WATER_COLOR, None);
                }
                let t = (p / max_p).clamp(0.0, 1.0);
                let fill = heat(t);
                let label = if p > 0.0 {
                    let txt_color = if t > 0.55 {
                        egui::Color32::from_gray(20)
                    } else {
                        egui::Color32::from_gray(220)
                    };
                    Some((format!("{:.0}", p * 100.0), txt_color))
                } else {
                    None
                };
                (fill, label)
            }
        }
    }
}

// ---- helpers & palette ----

const WATER_COLOR: egui::Color32 = egui::Color32::from_rgb(28, 40, 66);
const SHIP_COLOR: egui::Color32 = egui::Color32::from_rgb(70, 170, 110);
const MISS_COLOR: egui::Color32 = egui::Color32::from_rgb(60, 66, 82);
const HIT_COLOR: egui::Color32 = egui::Color32::from_rgb(220, 110, 50);
const SUNK_COLOR: egui::Color32 = egui::Color32::from_rgb(150, 40, 40);

fn remove_one(v: &mut Vec<usize>, x: usize) -> bool {
    if let Some(p) = v.iter().position(|&e| e == x) {
        v.remove(p);
        true
    } else {
        false
    }
}

fn fleet_str(ships: &[usize]) -> String {
    if ships.is_empty() {
        return "none".into();
    }
    let mut s = ships.to_vec();
    s.sort_unstable_by(|a, b| b.cmp(a));
    s.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")
}

fn group_digits(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::new();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// A1-style coordinate (column letter + 1-based row).
fn coord_name(x: usize, y: usize) -> String {
    let col = (b'A' + (x % 26) as u8) as char;
    format!("{}{}", col, y + 1)
}

/// Map a normalised value `t` in [0,1] to a blue→green→yellow→red heat colour.
pub fn heat(t: f64) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0) as f32;
    let low = egui::Color32::from_rgb(24, 36, 90);
    let mid = egui::Color32::from_rgb(40, 170, 120);
    let hi = egui::Color32::from_rgb(240, 210, 60);
    let top = egui::Color32::from_rgb(220, 55, 40);
    if t < 0.5 {
        lerp_color(low, mid, t / 0.5)
    } else if t < 0.8 {
        lerp_color(mid, hi, (t - 0.5) / 0.3)
    } else {
        lerp_color(hi, top, (t - 0.8) / 0.2)
    }
}

fn lerp_color(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    egui::Color32::from_rgb(f(a.r(), b.r()), f(a.g(), b.g()), f(a.b(), b.b()))
}
