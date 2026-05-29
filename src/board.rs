//! Core game geometry: the board configuration, ship placement enumeration,
//! and the placement-validity check that enforces the (optional) "no-touching"
//! gap rule.

use rand::seq::SliceRandom;
use rand::Rng;

/// Rules of the game. Everything here is runtime-configurable from the GUI.
#[derive(Clone, Debug)]
pub struct GameConfig {
    pub width: usize,
    pub height: usize,
    /// Ship lengths in the fleet, e.g. the classic `[5, 4, 3, 3, 2]`.
    pub ships: Vec<usize>,
    /// When true, ships may not touch — not even diagonally (a 1-tile moat
    /// around every ship). When false, ships may be placed edge-to-edge.
    pub gap_rule: bool,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            width: 10,
            height: 10,
            ships: vec![5, 4, 3, 3, 2],
            gap_rule: true,
        }
    }
}

impl GameConfig {
    #[inline]
    pub fn cells(&self) -> usize {
        self.width * self.height
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    #[inline]
    pub fn xy(&self, idx: usize) -> (usize, usize) {
        (idx % self.width, idx / self.width)
    }

    /// Largest dimension — the longest ship that could ever fit.
    pub fn max_dim(&self) -> usize {
        self.width.max(self.height)
    }
}

/// A concrete ship: its length and the cells it occupies.
#[derive(Clone, Debug)]
pub struct Ship {
    pub len: usize,
    pub cells: Vec<usize>,
}

/// A fully-known board (the "answer"). Used in known-board mode so the solver
/// can play against a real layout and shots can be auto-resolved.
#[derive(Clone, Debug)]
pub struct TrueBoard {
    /// `cell_ship[c] == Some(i)` when cell `c` belongs to `ships[i]`.
    pub cell_ship: Vec<Option<usize>>,
    pub ships: Vec<Ship>,
}

impl TrueBoard {
    /// Place the whole fleet at random, respecting overlap and the gap rule.
    /// Returns `None` if no valid arrangement was found within the attempt
    /// budget (e.g. too many ships for the board).
    pub fn random(cfg: &GameConfig, rng: &mut impl Rng) -> Option<TrueBoard> {
        // Place longest ships first — they are the hardest to fit.
        let mut order: Vec<usize> = cfg.ships.clone();
        order.sort_unstable_by(|a, b| b.cmp(a));

        'attempt: for _ in 0..3000 {
            let mut occ = vec![false; cfg.cells()];
            let mut cell_ship = vec![None; cfg.cells()];
            let mut ships: Vec<Ship> = Vec::with_capacity(order.len());

            for &len in &order {
                let candidates: Vec<Vec<usize>> = ship_placements(cfg, len)
                    .into_iter()
                    .filter(|p| can_place(cfg, &occ, p))
                    .collect();
                let Some(chosen) = candidates.choose(rng) else {
                    continue 'attempt; // dead end, retry the whole fleet
                };
                let ship_idx = ships.len();
                for &c in chosen {
                    occ[c] = true;
                    cell_ship[c] = Some(ship_idx);
                }
                ships.push(Ship {
                    len,
                    cells: chosen.clone(),
                });
            }
            return Some(TrueBoard { cell_ship, ships });
        }
        None
    }
}

/// Every in-bounds placement of a ship of `len`, in both orientations, as a
/// list of occupied cell indices. (Length-1 ships are emitted once.)
pub fn ship_placements(cfg: &GameConfig, len: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if len == 0 || len > cfg.max_dim() {
        return out;
    }
    // Horizontal placements.
    if len <= cfg.width {
        for y in 0..cfg.height {
            for x in 0..=cfg.width - len {
                out.push((0..len).map(|i| cfg.idx(x + i, y)).collect());
            }
        }
    }
    // Vertical placements (skip for len == 1 — identical to horizontal).
    if len > 1 && len <= cfg.height {
        for y in 0..=cfg.height - len {
            for x in 0..cfg.width {
                out.push((0..len).map(|i| cfg.idx(x, y + i)).collect());
            }
        }
    }
    out
}

/// Can these cells host a ship given what's already occupied?
///
/// `cells` is assumed not yet applied to `occ`, so any occupied neighbour is
/// necessarily a *different* ship — which is exactly what the gap rule forbids.
pub fn can_place(cfg: &GameConfig, occ: &[bool], cells: &[usize]) -> bool {
    for &c in cells {
        if occ[c] {
            return false;
        }
    }
    if cfg.gap_rule {
        for &c in cells {
            for nb in neighbors8(cfg, c) {
                if occ[nb] {
                    return false;
                }
            }
        }
    }
    true
}

/// The up-to-8 cells surrounding `cell` (orthogonal + diagonal).
pub fn neighbors8(cfg: &GameConfig, cell: usize) -> Vec<usize> {
    let (x, y) = cfg.xy(cell);
    let mut v = Vec::with_capacity(8);
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0 && ny >= 0 && (nx as usize) < cfg.width && (ny as usize) < cfg.height {
                v.push(cfg.idx(nx as usize, ny as usize));
            }
        }
    }
    v
}

/// The up-to-4 orthogonally-adjacent cells.
pub fn neighbors4(cfg: &GameConfig, cell: usize) -> Vec<usize> {
    let (x, y) = cfg.xy(cell);
    let mut v = Vec::with_capacity(4);
    const DIRS: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
    for (dx, dy) in DIRS {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if nx >= 0 && ny >= 0 && (nx as usize) < cfg.width && (ny as usize) < cfg.height {
            v.push(cfg.idx(nx as usize, ny as usize));
        }
    }
    v
}
