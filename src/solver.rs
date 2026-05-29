//! The probability engine.
//!
//! Given the evidence so far (misses, open hits, and sunk ships) the solver
//! computes, for every unknown cell, the probability that it hides part of a
//! remaining ship. The recommended shot is the unknown cell with the highest
//! probability.
//!
//! Two strategies, picked automatically (the "hybrid"):
//!
//! * **Exact** — enumerate *every* placement of the remaining fleet that is
//!   consistent with the evidence, counting how often each cell is occupied.
//!   `probability(cell) = occupied_count(cell) / total_consistent_configs`.
//!   This is the unbiased posterior. We fan the search out over all placements
//!   of the first (largest) ship with rayon and reduce the per-branch occupancy
//!   grids back together — exactly the map-reduce shape you sketched.
//!
//! * **Monte-Carlo** — when the consistent-configuration space is astronomically
//!   large (the opening on a full board), sample valid fleets at random in
//!   parallel and accumulate occupancy instead. Fast and scalable; approximate.
//!
//! Exact is used whenever it's feasible (small search space, or once hits
//! constrain things); Monte-Carlo covers the wide-open early game.

use crate::board::*;
use rayon::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellState {
    Unknown,
    Miss,
    /// A confirmed ship hit whose ship is not yet fully sunk.
    Hit,
    /// Part of a ship that has been completely sunk (fully known).
    Sunk,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Method {
    Exact,
    MonteCarlo,
    /// Nothing to compute (no ships left, or contradictory evidence).
    Trivial,
}

impl Method {
    pub fn label(self) -> &'static str {
        match self {
            Method::Exact => "exact (full enumeration)",
            Method::MonteCarlo => "Monte-Carlo sampling",
            Method::Trivial => "n/a",
        }
    }
}

pub struct SolveResult {
    /// Per-cell probability of containing a ship (0 for non-unknown cells).
    pub prob: Vec<f64>,
    /// Configs enumerated (exact) or samples accepted (Monte-Carlo).
    pub configs: u64,
    pub method: Method,
    /// Highest-probability unknown cell — the recommended shot.
    pub best_cell: Option<usize>,
    /// Probability at the recommended cell, i.e. the "confidence score".
    pub best_conf: f64,
}

/// If the *estimated* number of consistent configs is at or below this, run the
/// exact engine straight away.
const EXACT_ESTIMATE_GATE: u64 = 2_000_000;
/// Hard ceiling on configs the exact engine will enumerate before bailing to
/// Monte-Carlo. Bounds worst-case latency.
const EXACT_HARD_CAP: u64 = 4_000_000;
/// Monte-Carlo attempt budget (across all threads). Enough samples for a smooth
/// heatmap while staying well under a frame in release builds.
const MC_ATTEMPTS: u64 = 40_000;

/// Compute the probability heatmap for the current evidence.
pub fn solve(cfg: &GameConfig, states: &[CellState], remaining: &[usize]) -> SolveResult {
    let n = cfg.cells();
    let mut prob = vec![0.0; n];

    if remaining.is_empty() {
        return SolveResult {
            prob,
            configs: 0,
            method: Method::Trivial,
            best_cell: None,
            best_conf: 0.0,
        };
    }

    // Cells a remaining ship may never occupy: misses, already-sunk cells, and
    // (under the gap rule) the moat around sunk ships.
    let mut blocked = vec![false; n];
    for (c, &s) in states.iter().enumerate() {
        if matches!(s, CellState::Miss | CellState::Sunk) {
            blocked[c] = true;
        }
    }
    if cfg.gap_rule {
        for c in 0..n {
            if states[c] == CellState::Sunk {
                for nb in neighbors8(cfg, c) {
                    blocked[nb] = true;
                }
            }
        }
    }
    // Open hits must remain coverable even if the evidence looks contradictory.
    for c in 0..n {
        if states[c] == CellState::Hit {
            blocked[c] = false;
        }
    }

    // Remaining ships, longest first (better pruning and coarser fan-out).
    let mut lens: Vec<usize> = remaining.to_vec();
    lens.sort_unstable_by(|a, b| b.cmp(a));

    // Static placements per ship: in-bounds and clear of blocked cells. The
    // dynamic overlap/gap checks happen during the search.
    let placements: Vec<Vec<Vec<usize>>> = lens
        .iter()
        .map(|&len| {
            ship_placements(cfg, len)
                .into_iter()
                .filter(|p| p.iter().all(|&c| !blocked[c]))
                .collect()
        })
        .collect();

    // Any ship with nowhere to go ⇒ evidence is unsatisfiable.
    if placements.iter().any(|p| p.is_empty()) {
        return SolveResult {
            prob,
            configs: 0,
            method: Method::Trivial,
            best_cell: None,
            best_conf: 0.0,
        };
    }

    let hits: Vec<usize> = (0..n).filter(|&c| states[c] == CellState::Hit).collect();

    // Rough upper bound on the search size (ignores mutual exclusion).
    let estimate = placements
        .iter()
        .fold(1u64, |acc, p| acc.saturating_mul(p.len() as u64));

    // Use exact when the space is small, or whenever hits constrain it (then the
    // consistent set is usually tiny). Otherwise sample.
    let prefer_exact = estimate <= EXACT_ESTIMATE_GATE || !hits.is_empty();

    let (counts, total, method) = if prefer_exact {
        match exact(cfg, &placements, &hits, &lens, EXACT_HARD_CAP) {
            Some((grid, t)) => (grid, t, Method::Exact),
            None => {
                let (grid, t) = monte_carlo(cfg, &placements, &hits, MC_ATTEMPTS);
                (grid, t, Method::MonteCarlo)
            }
        }
    } else {
        let (grid, t) = monte_carlo(cfg, &placements, &hits, MC_ATTEMPTS);
        (grid, t, Method::MonteCarlo)
    };

    if total > 0 {
        for c in 0..n {
            if states[c] == CellState::Unknown {
                prob[c] = counts[c] as f64 / total as f64;
            }
        }
    }

    let mut best_cell = None;
    let mut best_conf = 0.0f64;
    for c in 0..n {
        if states[c] == CellState::Unknown && prob[c] > best_conf {
            best_conf = prob[c];
            best_cell = Some(c);
        }
    }

    SolveResult {
        prob,
        configs: total,
        method,
        best_cell,
        best_conf,
    }
}

/// Exact enumeration. Returns `(occupancy_counts, total_configs)`, or `None` if
/// the search exceeded `cap` configs (caller should fall back to Monte-Carlo).
///
/// The fan-out: every placement of ship 0 becomes an independent rayon task that
/// recursively places the rest; each task accumulates its own occupancy grid,
/// and the grids are summed in the reduce.
fn exact(
    cfg: &GameConfig,
    placements: &[Vec<Vec<usize>>],
    hits: &[usize],
    lens: &[usize],
    cap: u64,
) -> Option<(Vec<u64>, u64)> {
    let n = cfg.cells();
    let counter = AtomicU64::new(0); // total configs across all branches
    let aborted = AtomicBool::new(false);

    // rem_cells[i] = total cells in ships i.. — used to prune branches that can
    // no longer cover all open hits.
    let mut rem_cells = vec![0usize; lens.len() + 1];
    for i in (0..lens.len()).rev() {
        rem_cells[i] = rem_cells[i + 1] + lens[i];
    }

    let (grid, total) = placements[0]
        .par_iter()
        .map(|p0| {
            if aborted.load(Ordering::Relaxed) {
                return (vec![0u64; n], 0u64);
            }
            let mut occ = vec![false; n];
            let mut placed: Vec<usize> = Vec::with_capacity(64);
            let mut local = vec![0u64; n];
            let mut local_count = 0u64;

            for &c in p0 {
                occ[c] = true;
                placed.push(c);
            }
            backtrack(
                cfg,
                placements,
                1,
                &mut occ,
                &mut placed,
                &mut local,
                &mut local_count,
                hits,
                &rem_cells,
                &counter,
                &aborted,
                cap,
            );
            (local, local_count)
        })
        .reduce(
            || (vec![0u64; n], 0u64),
            |mut a, b| {
                for i in 0..n {
                    a.0[i] += b.0[i];
                }
                a.1 += b.1;
                a
            },
        );

    if aborted.load(Ordering::Relaxed) {
        None
    } else {
        Some((grid, total))
    }
}

#[allow(clippy::too_many_arguments)]
fn backtrack(
    cfg: &GameConfig,
    placements: &[Vec<Vec<usize>>],
    idx: usize,
    occ: &mut [bool],
    placed: &mut Vec<usize>,
    local: &mut [u64],
    local_count: &mut u64,
    hits: &[usize],
    rem_cells: &[usize],
    counter: &AtomicU64,
    aborted: &AtomicBool,
    cap: u64,
) {
    if aborted.load(Ordering::Relaxed) {
        return;
    }

    if idx == placements.len() {
        // A full fleet is placed — accept only if every open hit is covered.
        if hits.iter().any(|&h| !occ[h]) {
            return;
        }
        let prev = counter.fetch_add(1, Ordering::Relaxed);
        if prev + 1 > cap {
            aborted.store(true, Ordering::Relaxed);
            return;
        }
        for &c in placed.iter() {
            local[c] += 1;
        }
        *local_count += 1;
        return;
    }

    // Prune: if more open hits are uncovered than the remaining ships can
    // possibly cover, this branch is hopeless.
    let uncovered = hits.iter().filter(|&&h| !occ[h]).count();
    if uncovered > rem_cells[idx] {
        return;
    }

    for p in &placements[idx] {
        if can_place(cfg, occ, p) {
            for &c in p {
                occ[c] = true;
                placed.push(c);
            }
            backtrack(
                cfg,
                placements,
                idx + 1,
                occ,
                placed,
                local,
                local_count,
                hits,
                rem_cells,
                counter,
                aborted,
                cap,
            );
            for &c in p {
                occ[c] = false;
            }
            placed.truncate(placed.len() - p.len());

            if aborted.load(Ordering::Relaxed) {
                return;
            }
        }
    }
}

/// Monte-Carlo occupancy estimate. Each thread repeatedly places the fleet by
/// choosing, for every ship in turn, a uniformly-random *currently-valid*
/// placement (reservoir sampling, single pass, no allocation). Samples that
/// fail to cover all open hits are rejected. Returns `(counts, accepted)`.
fn monte_carlo(
    cfg: &GameConfig,
    placements: &[Vec<Vec<usize>>],
    hits: &[usize],
    attempts: u64,
) -> (Vec<u64>, u64) {
    let n = cfg.cells();
    let batches = (rayon::current_num_threads().max(1)) as u64;
    let per = (attempts / batches).max(1);

    let (mut grid, mut total) = (0..batches)
        .into_par_iter()
        .map(|b| {
            // Deterministic per-batch seed (no wall-clock — keeps runs stable
            // and avoids the sandbox's clock restrictions).
            let seed = 0x9E3779B97F4A7C15u64
                ^ b.wrapping_mul(0x100000001B3)
                ^ (hits.len() as u64).wrapping_mul(0x2545F4914F6CDD1D)
                ^ (placements.len() as u64).wrapping_mul(31);
            let mut rng = StdRng::seed_from_u64(seed);

            let mut local = vec![0u64; n];
            let mut count = 0u64;
            let mut occ = vec![false; n];
            let mut placed: Vec<usize> = Vec::with_capacity(64);

            for _ in 0..per {
                for &c in &placed {
                    occ[c] = false;
                }
                placed.clear();

                let mut ok = true;
                for ship in placements {
                    // Reservoir-sample one valid placement in a single pass.
                    let mut chosen: Option<&Vec<usize>> = None;
                    let mut seen = 0u32;
                    for p in ship {
                        if can_place(cfg, &occ, p) {
                            seen += 1;
                            if rng.gen_range(0..seen) == 0 {
                                chosen = Some(p);
                            }
                        }
                    }
                    match chosen {
                        Some(p) => {
                            for &c in p {
                                occ[c] = true;
                                placed.push(c);
                            }
                        }
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    continue;
                }
                if hits.iter().any(|&h| !occ[h]) {
                    continue; // doesn't explain the hits
                }
                for &c in &placed {
                    local[c] += 1;
                }
                count += 1;
            }
            (local, count)
        })
        .reduce(
            || (vec![0u64; n], 0u64),
            |mut a, b| {
                for i in 0..n {
                    a.0[i] += b.0[i];
                }
                a.1 += b.1;
                a
            },
        );

    // Safety net: if rejection sampling never explained the hits (rare — hits
    // normally route to the exact engine), fall back to a crude per-ship
    // density, biased toward placements that touch a hit, so we still return a
    // usable, non-empty heatmap.
    if total == 0 {
        for ship in placements {
            for p in ship {
                let weight = if hits.iter().any(|&h| p.contains(&h)) { 4 } else { 1 };
                for &c in p {
                    grid[c] += weight;
                }
                total += weight;
            }
        }
    }

    (grid, total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Resolve a shot against a known board and update evidence, mirroring the
    /// GUI's known-board logic. Returns true on a hit.
    fn resolve(
        board: &TrueBoard,
        states: &mut [CellState],
        remaining: &mut Vec<usize>,
        cell: usize,
    ) -> bool {
        if let Some(si) = board.cell_ship[cell] {
            states[cell] = CellState::Hit;
            let ship = &board.ships[si];
            if ship
                .cells
                .iter()
                .all(|&c| matches!(states[c], CellState::Hit | CellState::Sunk))
            {
                for &c in &ship.cells {
                    states[c] = CellState::Sunk;
                }
                if let Some(p) = remaining.iter().position(|&l| l == ship.len) {
                    remaining.remove(p);
                }
            }
            true
        } else {
            states[cell] = CellState::Miss;
            false
        }
    }

    /// Play a full game greedily against `board`; returns the shot count.
    fn play_out(cfg: &GameConfig, board: &TrueBoard) -> u32 {
        let n = cfg.cells();
        let mut states = vec![CellState::Unknown; n];
        let mut remaining = cfg.ships.clone();
        let mut shots = 0u32;
        while !remaining.is_empty() && (shots as usize) < n {
            let res = solve(cfg, &states, &remaining);
            let bc = res.best_cell.expect("solver must recommend a cell while ships remain");
            assert_eq!(states[bc], CellState::Unknown, "recommended an already-shot cell");
            resolve(board, &mut states, &mut remaining, bc);
            shots += 1;
        }
        assert!(remaining.is_empty(), "game did not finish: {:?}", remaining);
        shots
    }

    #[test]
    fn misses_get_zero_probability_and_are_never_recommended() {
        let cfg = GameConfig::default();
        let mut states = vec![CellState::Unknown; cfg.cells()];
        let m = cfg.idx(3, 3);
        states[m] = CellState::Miss;
        let res = solve(&cfg, &states, &cfg.ships);
        assert_eq!(res.prob[m], 0.0);
        assert_ne!(res.best_cell, Some(m));
    }

    #[test]
    fn probabilities_are_in_range() {
        let cfg = GameConfig::default();
        let mut states = vec![CellState::Unknown; cfg.cells()];
        states[cfg.idx(4, 4)] = CellState::Hit; // forces the exact engine
        let res = solve(&cfg, &states, &cfg.ships);
        for &p in &res.prob {
            assert!((0.0..=1.0).contains(&p), "probability out of range: {p}");
        }
        // Cells orthogonally adjacent to a lone hit should be the hot spots.
        let neigh = neighbors4(&cfg, cfg.idx(4, 4));
        let best = res.best_cell.unwrap();
        assert!(neigh.contains(&best), "best shot should hug the hit");
    }

    #[test]
    fn exact_and_monte_carlo_roughly_agree() {
        // Small board where exact is cheap and MC accepts nearly everything.
        let cfg = GameConfig {
            width: 6,
            height: 6,
            ships: vec![3, 2],
            gap_rule: true,
        };
        let n = cfg.cells();
        let empty_hits: Vec<usize> = Vec::new();
        let lens = {
            let mut l = cfg.ships.clone();
            l.sort_unstable_by(|a, b| b.cmp(a));
            l
        };
        let placements: Vec<Vec<Vec<usize>>> = lens
            .iter()
            .map(|&len| ship_placements(&cfg, len))
            .collect();

        let (eg, et) = exact(&cfg, &placements, &empty_hits, &lens, 100_000_000).unwrap();
        let (mg, mt) = monte_carlo(&cfg, &placements, &empty_hits, 200_000);
        assert!(et > 0 && mt > 0);

        // Compare normalized per-cell densities; sequential-placement MC is only
        // approximate, so allow a generous tolerance.
        let mut max_diff = 0.0_f64;
        for c in 0..n {
            let pe = eg[c] as f64 / et as f64;
            let pm = mg[c] as f64 / mt as f64;
            max_diff = max_diff.max((pe - pm).abs());
        }
        assert!(max_diff < 0.15, "exact vs MC diverged: max_diff={max_diff:.3}");
    }

    #[test]
    fn solves_small_board_quickly() {
        let cfg = GameConfig {
            width: 6,
            height: 6,
            ships: vec![3, 2],
            gap_rule: true,
        };
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..8 {
            let board = TrueBoard::random(&cfg, &mut rng).unwrap();
            let shots = play_out(&cfg, &board);
            assert!(shots <= cfg.cells() as u32);
        }
    }

    /// Realistic benchmark on the standard fleet — reports average shots to win.
    /// Heavier; run with: `cargo test --release -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn benchmark_standard_fleet() {
        let cfg = GameConfig::default();
        let mut rng = StdRng::seed_from_u64(2024);
        let games = 30;
        let mut total = 0u32;
        let mut worst = 0u32;
        for _ in 0..games {
            let board = TrueBoard::random(&cfg, &mut rng).unwrap();
            let shots = play_out(&cfg, &board);
            total += shots;
            worst = worst.max(shots);
        }
        let avg = total as f64 / games as f64;
        println!("standard 10x10 fleet: avg {avg:.1} shots, worst {worst} (random firing ≈ 96)");
        assert!(avg < 65.0, "solver too weak: avg {avg:.1}");
    }
}
