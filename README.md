# battleship-solver

A probability-based Battleship solver with a live GUI heatmap and a parallel
hybrid engine.

For each move the solver considers every way the remaining fleet could be
arranged given what's been revealed so far, then shades each unknown cell by how
often a ship lands there. The brightest cell is the recommended shot, and its
probability is the displayed confidence score.

On the standard 10×10 fleet with the no-touching rule it sinks everything in
**~38 shots on average** (worst case ~48 in benchmarking), versus ~96 for random
firing.

## Running

egui is much faster in release mode — always run it that way:

```bash
cargo run --release
```

The project's [Cargo configuration](.cargo/config.toml) automatically sets
`LIBGL_ALWAYS_SOFTWARE=true` and `GALLIUM_DRIVER=llvmpipe`, so no extra launch
arguments are needed. This uses Mesa's CPU renderer to avoid WSL's `libEGL` and
Zink startup warnings. Existing environment variables can override these
defaults.

## How it works

The engine builds the exact posterior over ship locations by configuration
counting, and picks one of two strategies automatically (the "hybrid"):

- **Exact enumeration.** Enumerate *every* placement of the remaining fleet
  consistent with the evidence (misses are empty, every open hit is covered,
  sunk ships are fixed, the gap rule holds), counting how often each cell is
  occupied. `P(cell) = occupied_count(cell) / total_consistent_configs` — the
  unbiased marginal. The search **fans out over all placements of the first
  (largest) ship with `rayon`** and recurses for the rest; each branch keeps its
  own occupancy grid and the grids are summed in the reduce. That's the
  map-reduce shape from the brief: place the next ship at every legal spot,
  recurse, then map the results into a confidence score.

- **Monte-Carlo sampling.** In the wide-open early game the consistent-config
  space is astronomically large, so instead of enumerating it we sample valid
  fleets at random in parallel and accumulate occupancy. Fast and scalable;
  approximate.

Exact runs whenever it's feasible (a small search space, or once hits constrain
things — which is most of the game); Monte-Carlo covers the opening. The
crossover is automatic, based on a cheap upper-bound estimate of the search
size, with a hard cap that falls back to sampling if exact enumeration would run
too long.

## Modes

- **Known board** — place your fleet by hand (click to place, right-click or `R`
  to rotate) or hit **Randomize Board**. Then watch the solver play: **Fire
  Best** (`Space`), **Auto-play** (animated, adjustable speed), or **Solve
  Instantly**. Shots auto-resolve against the real board, including sinking.
- **Assistant** — for solving a real game against a human opponent, where no
  board is known. Fire at the recommended cell, then mark the result yourself:
  left-click cycles Unknown → Miss → Hit, right-click sets a hit, and the
  **Sink tool** turns a contiguous run of hits into a sunk ship.

## Configuration

Board width/height, the ship list, and the no-touching gap rule are all editable
in the **Configuration** panel and apply on **Apply & New Game**. Defaults:
10×10, ships `5,4,3,3,2`, no-touching on.

## Project layout

| File | Responsibility |
|------|----------------|
| `src/board.rs`  | Geometry, placement enumeration, the gap-rule validity check, random fleets |
| `src/solver.rs` | The hybrid probability engine + tests |
| `src/app.rs`    | egui front-end: placement, heatmap, play modes |
| `src/main.rs`   | Window setup / entry point |

## Tests

```bash
cargo test                                   # fast correctness invariants
cargo test --release -- --ignored --nocapture  # average-shots benchmark
```
