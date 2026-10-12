# Multi-site coordinate swap: J-head vs Logit-head

Experiment 2's sites and scope (every position of every selected head), with each
site's coordinates clamped to the swapped clean-pass ones and the frame built through
J^z (J) or W_O (Logit). The swap's size follows
each lens's coordinates, so edit norms differ between lenses (last line per table).
Cells are successes out of the task's trials. `J` / `L` name the
lens; `sel` is the lens that chose the sites, `edit` the lens that built the edit.
Differences are paired per trial, J minus Logit. Random sites are shared by both
lenses; bottom sites use each lens's own
loading; random direction is a same-norm random edit at the lens's own selected sites.

probe-swap: median active count J 17, Logit 8; mean J/Logit top-set overlap top_1: 0.3, top_8: 3.3, top_16: 6.8, top_32: 14.9, top_64: 40.2, active: 5.4
verbal-report: median active count J 16, Logit 13; mean J/Logit top-set overlap top_1: 0.1, top_8: 3.2, top_16: 5.8, top_32: 12.4, top_64: 37.1, active: 5.4

## probe-swap, alpha = 1 (53 trials)

| sites | J sel+edit | L sel+edit | end-to-end diff | random sites J / L | bottom J / L | random direction J / L | direction diff at J sites | selection diff, J edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | +0.000 | 1 / 1 | 1 / 1 | 1 / 1 | +0.000 | +0.000 |
| 8 | 2 | 2 | +0.000 | 1 / 1 | 1 / 1 | 1 / 1 | +0.000 | -0.019 |
| 16 | 3 | 2 | +0.019 | 1 / 1 | 1 / 1 | 1 / 1 | +0.019 | +0.000 |
| 32 | 4 | 2 | +0.038 | 1 / 1 | 1 / 1 | 1 / 1 | +0.038 | +0.019 |
| 64 | 7 | 2 | +0.094 | 1 / 1 | 1 / 1 | 1 / 1 | +0.075 | +0.075 |
| active | 5 | 2 | +0.057 | 1 / 1 | 1 / 1 | 1 / 1 | +0.057 | +0.038 |
| layer-wide | 27 | 10 | +0.321 | | | 0 / 0 | | |
| layer-wide, band 12-28 | 25 | 8 | | | | 0 / 0 | | |

Mean KL from clean, top_64 J / L: 0.259 / 0.067; layer-wide J / L: 2.470 / 0.794; band J / L: 2.675 / 0.581.
Mean total edit norm at each lens's own top sites, J / L: 1: 1.8 / 2.4, 8: 5.5 / 5.9, 16: 7.0 / 7.7, 32: 8.4 / 9.4, 64: 11.7 / 12.2.

## probe-swap, alpha = 2 (53 trials)

| sites | J sel+edit | L sel+edit | end-to-end diff | random sites J / L | bottom J / L | random direction J / L | direction diff at J sites | selection diff, J edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 3 | -0.019 | 1 / 1 | 1 / 1 | 1 / 1 | +0.000 | -0.038 |
| 8 | 14 | 7 | +0.132 | 3 / 1 | 1 / 1 | 1 / 1 | +0.132 | -0.019 |
| 16 | 20 | 9 | +0.208 | 3 / 1 | 1 / 1 | 1 / 1 | +0.226 | +0.038 |
| 32 | 24 | 11 | +0.245 | 5 / 2 | 1 / 1 | 1 / 1 | +0.264 | +0.038 |
| 64 | 25 | 13 | +0.226 | 8 / 3 | 1 / 1 | 1 / 1 | +0.245 | +0.038 |
| active | 19 | 9 | +0.189 | 3 / 1 | 1 / 1 | 1 / 1 | +0.208 | +0.075 |
| layer-wide | 20 | 19 | +0.019 | | | 0 / 0 | | |
| layer-wide, band 12-28 | 20 | 17 | | | | 1 / 1 | | |

Mean KL from clean, top_64 J / L: 2.608 / 0.864; layer-wide J / L: 6.361 / 4.124; band J / L: 6.393 / 3.435.
Mean total edit norm at each lens's own top sites, J / L: 1: 3.6 / 4.8, 8: 10.9 / 11.8, 16: 13.9 / 15.4, 32: 16.6 / 18.8, 64: 23.2 / 24.4.

## verbal-report, alpha = 1 (140 trials)

| sites | J sel+edit | L sel+edit | end-to-end diff | random sites J / L | bottom J / L | random direction J / L | direction diff at J sites | selection diff, J edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 1 | +0.007 | 0 / 0 | 0 / 0 | 0 / 0 | +0.014 | -0.007 |
| 8 | 18 | 8 | +0.071 | 0 / 0 | 0 / 0 | 0 / 0 | +0.100 | +0.079 |
| 16 | 29 | 15 | +0.100 | 0 / 0 | 0 / 0 | 0 / 0 | +0.179 | +0.136 |
| 32 | 58 | 12 | +0.329 | 4 / 0 | 0 / 0 | 0 / 0 | +0.386 | +0.400 |
| 64 | 46 | 11 | +0.250 | 2 / 0 | 0 / 0 | 0 / 0 | +0.286 | +0.221 |
| active | 73 | 9 | +0.457 | 0 / 0 | 0 / 0 | 0 / 0 | +0.500 | +0.457 |
| layer-wide | 30 | 10 | +0.143 | | | 0 / 0 | | |
| layer-wide, band 12-28 | 27 | 6 | | | | 0 / 0 | | |

Mean KL from clean, top_64 J / L: 0.249 / 0.322; layer-wide J / L: 0.777 / 0.353; band J / L: 0.628 / 0.226.
Mean total edit norm at each lens's own top sites, J / L: 1: 4.4 / 9.0, 8: 10.3 / 14.0, 16: 11.4 / 15.3, 32: 14.9 / 16.6, 64: 18.1 / 18.7.

## verbal-report, alpha = 2 (140 trials)

| sites | J sel+edit | L sel+edit | end-to-end diff | random sites J / L | bottom J / L | random direction J / L | direction diff at J sites | selection diff, J edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7 | 5 | +0.014 | 0 / 0 | 0 / 0 | 0 / 0 | +0.036 | -0.029 |
| 8 | 75 | 55 | +0.143 | 1 / 1 | 0 / 0 | 0 / 0 | +0.357 | +0.214 |
| 16 | 83 | 66 | +0.121 | 2 / 2 | 0 / 0 | 0 / 0 | +0.400 | +0.136 |
| 32 | 88 | 64 | +0.171 | 12 / 2 | 0 / 0 | 0 / 0 | +0.400 | +0.307 |
| 64 | 95 | 71 | +0.171 | 19 / 7 | 0 / 0 | 0 / 0 | +0.400 | +0.393 |
| active | 90 | 56 | +0.243 | 3 / 2 | 0 / 0 | 0 / 0 | +0.443 | +0.336 |
| layer-wide | 40 | 39 | +0.007 | | | 0 / 0 | | |
| layer-wide, band 12-28 | 39 | 37 | | | | 0 / 0 | | |

Mean KL from clean, top_64 J / L: 1.769 / 1.503; layer-wide J / L: 3.705 / 2.110; band J / L: 3.118 / 1.518.
Mean total edit norm at each lens's own top sites, J / L: 1: 8.8 / 18.1, 8: 20.7 / 28.0, 16: 22.9 / 30.6, 32: 29.9 / 33.3, 64: 36.3 / 37.5.

## Layer-wide references, scored as Gurnee et al. score theirs

Target rank after the swap, top-1 and top-5, over all trials and over the trials whose
target was outside the clean top 10 (the paper swaps in only such verbal-report targets).
`band` is block outputs 12-28, the paper's workspace depth.

| task | alpha | scope | lens | top-1 | top-5 | top-1, outside top-10 | top-5, outside top-10 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| probe-swap | 1 | 0-30 | Jacobian lens | 27/53 | 43/53 | 14/31 | 21/31 |
| probe-swap | 1 | 0-30 | unembedding | 10/53 | 34/53 | 4/31 | 13/31 |
| probe-swap | 1 | band | Jacobian lens | 25/53 | 44/53 | 14/31 | 23/31 |
| probe-swap | 1 | band | unembedding | 8/53 | 31/53 | 5/31 | 10/31 |
| probe-swap | 2 | 0-30 | Jacobian lens | 20/53 | 42/53 | 10/31 | 20/31 |
| probe-swap | 2 | 0-30 | unembedding | 19/53 | 38/53 | 8/31 | 16/31 |
| probe-swap | 2 | band | Jacobian lens | 20/53 | 42/53 | 10/31 | 21/31 |
| probe-swap | 2 | band | unembedding | 17/53 | 39/53 | 7/31 | 17/31 |
| verbal-report | 1 | 0-30 | Jacobian lens | 30/140 | 39/140 | 30/131 | 39/131 |
| verbal-report | 1 | 0-30 | unembedding | 10/140 | 30/140 | 10/131 | 30/131 |
| verbal-report | 1 | band | Jacobian lens | 27/140 | 35/140 | 27/131 | 35/131 |
| verbal-report | 1 | band | unembedding | 6/140 | 21/140 | 6/131 | 21/131 |
| verbal-report | 2 | 0-30 | Jacobian lens | 40/140 | 44/140 | 40/131 | 44/131 |
| verbal-report | 2 | 0-30 | unembedding | 39/140 | 42/140 | 39/131 | 42/131 |
| verbal-report | 2 | band | Jacobian lens | 39/140 | 41/140 | 39/131 | 41/131 |
| verbal-report | 2 | band | unembedding | 37/140 | 42/140 | 37/131 | 42/131 |

Full cells (all selection x edit combinations, KL, edit norms): `summary.json`.
