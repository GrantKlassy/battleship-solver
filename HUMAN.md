# battleship-solver

A solver for the game "battleship"

A fancy GUI is created to let you place ships
But in the auto mode, a random board is generated.

Then, when deciding where to shoot, we go through all permutations
The permutations checks the ships left to find, the current hits,
The "one tile gap" between ships check. etc etc.

Then once we itterate through all possible boards, we give a confidence score

I did this once before but had a hard time running in parallel
I think some map reduce would work well

Like. Imagine a 9x9 board and we know where a single "3 long" ship is

Then for the next move, the solver should place a 4 long ship at all possible locations
And recursively fan out from there. Then map the results and confidence.

This is all written in rust.
