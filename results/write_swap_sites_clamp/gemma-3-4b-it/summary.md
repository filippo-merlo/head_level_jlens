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

probe-swap: median active count J 46, Logit 38; mean J/Logit top-set overlap top_1: 0.8, top_8: 5.8, top_16: 11.4, top_32: 21.7, top_64: 41.6, active: 28.3
verbal-report: median active count J 22, Logit 86; mean J/Logit top-set overlap top_1: 0.0, top_8: 1.0, top_16: 4.1, top_32: 8.8, top_64: 21.6, active: 13.6

## probe-swap, alpha = 1 (46 trials)

| sites | J sel+edit | L sel+edit | end-to-end diff | random sites J / L | bottom J / L | random direction J / L | direction diff at J sites | selection diff, J edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14 | 16 | -0.043 | 2 / 2 | 1 / 1 | 1 / 1 | +0.000 | -0.043 |
| 8 | 21 | 21 | +0.000 | 2 / 2 | 1 / 1 | 1 / 1 | +0.022 | -0.022 |
| 16 | 23 | 23 | +0.000 | 3 / 2 | 1 / 1 | 1 / 1 | -0.022 | +0.000 |
| 32 | 26 | 25 | +0.022 | 4 / 3 | 1 / 2 | 1 / 1 | +0.043 | -0.022 |
| 64 | 26 | 24 | +0.043 | 7 / 6 | 1 / 2 | 1 / 1 | +0.043 | +0.022 |
| active | 26 | 25 | +0.022 | 7 / 6 | 1 / 2 | 1 / 1 | +0.043 | -0.022 |
| layer-wide | 22 | 11 | +0.239 | | | 3 / 3 | | |
| layer-wide, band 13-30 | 25 | 8 | | | | 1 / 1 | | |

Mean KL from clean, top_64 J / L: 7.625 / 6.953; layer-wide J / L: 7.369 / 4.759; band J / L: 7.218 / 2.135.
Mean total edit norm at each lens's own top sites, J / L: 1: 16.6 / 17.4, 8: 26.8 / 31.3, 16: 29.0 / 38.0, 32: 36.7 / 64.0, 64: 80.3 / 117.5.

## probe-swap, alpha = 2 (46 trials)

| sites | J sel+edit | L sel+edit | end-to-end diff | random sites J / L | bottom J / L | random direction J / L | direction diff at J sites | selection diff, J edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 17 | 17 | +0.000 | 3 / 3 | 1 / 1 | 1 / 1 | +0.022 | -0.022 |
| 8 | 25 | 23 | +0.043 | 3 / 3 | 1 / 2 | 1 / 1 | +0.022 | +0.022 |
| 16 | 23 | 22 | +0.022 | 3 / 3 | 0 / 2 | 1 / 1 | +0.000 | +0.043 |
| 32 | 23 | 22 | +0.022 | 7 / 6 | 0 / 2 | 1 / 1 | +0.000 | +0.022 |
| 64 | 24 | 22 | +0.043 | 10 / 11 | 0 / 1 | 1 / 1 | +0.000 | -0.022 |
| active | 23 | 22 | +0.022 | 10 / 10 | 0 / 1 | 1 / 1 | +0.000 | +0.022 |
| layer-wide | 7 | 11 | -0.087 | | | 2 / 2 | | |
| layer-wide, band 13-30 | 12 | 12 | | | | 2 / 2 | | |

Mean KL from clean, top_64 J / L: 15.118 / 14.239; layer-wide J / L: 14.674 / 12.297; band J / L: 13.955 / 9.499.
Mean total edit norm at each lens's own top sites, J / L: 1: 33.3 / 34.8, 8: 52.9 / 61.8, 16: 57.3 / 75.1, 32: 73.2 / 127.3, 64: 160.3 / 234.4.

## verbal-report, alpha = 1 (140 trials)

| sites | J sel+edit | L sel+edit | end-to-end diff | random sites J / L | bottom J / L | random direction J / L | direction diff at J sites | selection diff, J edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | +0.000 | 0 / 0 | 0 / 0 | 0 / 0 | +0.000 | +0.000 |
| 8 | 7 | 5 | +0.014 | 2 / 4 | 0 / 0 | 0 / 0 | -0.021 | +0.050 |
| 16 | 9 | 79 | -0.500 | 1 / 8 | 0 / 0 | 0 / 0 | -0.114 | +0.000 |
| 32 | 14 | 135 | -0.864 | 1 / 31 | 0 / 0 | 0 / 0 | -0.300 | +0.007 |
| 64 | 44 | 138 | -0.671 | 1 / 80 | 0 / 0 | 0 / 0 | -0.521 | +0.279 |
| active | 13 | 140 | -0.907 | 0 / 13 | 0 / 0 | 0 / 0 | -0.207 | +0.050 |
| layer-wide | 79 | 112 | -0.236 | | | 0 / 0 | | |
| layer-wide, band 13-30 | 0 | 37 | | | | 0 / 1 | | |

Mean KL from clean, top_64 J / L: 1.980 / 19.064; layer-wide J / L: 4.980 / 9.658; band J / L: 1.429 / 4.928.
Mean total edit norm at each lens's own top sites, J / L: 1: 5.2 / 200.1, 8: 28.6 / 467.3, 16: 65.6 / 517.7, 32: 117.5 / 525.7, 64: 200.5 / 649.5.

## verbal-report, alpha = 2 (140 trials)

| sites | J sel+edit | L sel+edit | end-to-end diff | random sites J / L | bottom J / L | random direction J / L | direction diff at J sites | selection diff, J edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | +0.000 | 0 / 0 | 0 / 0 | 0 / 0 | -0.014 | +0.000 |
| 8 | 12 | 83 | -0.507 | 2 / 12 | 0 / 0 | 0 / 0 | -0.050 | +0.086 |
| 16 | 33 | 131 | -0.700 | 3 / 26 | 0 / 0 | 0 / 0 | -0.264 | +0.121 |
| 32 | 58 | 140 | -0.586 | 2 / 74 | 0 / 0 | 0 / 0 | -0.407 | +0.121 |
| 64 | 109 | 131 | -0.157 | 7 / 122 | 0 / 0 | 0 / 0 | -0.221 | +0.750 |
| active | 42 | 134 | -0.657 | 2 / 45 | 0 / 0 | 0 / 0 | -0.464 | +0.229 |
| layer-wide | 104 | 132 | -0.200 | | | 0 / 0 | | |
| layer-wide, band 13-30 | 16 | 80 | | | | 0 / 1 | | |

Mean KL from clean, top_64 J / L: 7.501 / 31.987; layer-wide J / L: 15.158 / 31.452; band J / L: 4.430 / 12.952.
Mean total edit norm at each lens's own top sites, J / L: 1: 10.3 / 400.2, 8: 57.1 / 935.0, 16: 131.0 / 1036.6, 32: 234.8 / 1053.5, 64: 400.8 / 1309.7.

## Layer-wide references, scored as Gurnee et al. score theirs

Target rank after the swap, top-1 and top-5, over all trials and over the trials whose
target was outside the clean top 10 (the paper swaps in only such verbal-report targets).
`band` is block outputs 13-30, the paper's workspace depth.

| task | alpha | scope | lens | top-1 | top-5 | top-1, outside top-10 | top-5, outside top-10 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| probe-swap | 1 | 0-32 | Jacobian lens | 22/46 | 37/46 | 17/33 | 27/33 |
| probe-swap | 1 | 0-32 | unembedding | 11/46 | 31/46 | 6/33 | 19/33 |
| probe-swap | 1 | band | Jacobian lens | 25/46 | 35/46 | 19/33 | 25/33 |
| probe-swap | 1 | band | unembedding | 8/46 | 25/46 | 5/33 | 13/33 |
| probe-swap | 2 | 0-32 | Jacobian lens | 7/46 | 22/46 | 5/33 | 15/33 |
| probe-swap | 2 | 0-32 | unembedding | 11/46 | 26/46 | 7/33 | 17/33 |
| probe-swap | 2 | band | Jacobian lens | 12/46 | 28/46 | 7/33 | 21/33 |
| probe-swap | 2 | band | unembedding | 12/46 | 26/46 | 8/33 | 16/33 |
| verbal-report | 1 | 0-32 | Jacobian lens | 79/140 | 110/140 | 70/102 | 89/102 |
| verbal-report | 1 | 0-32 | unembedding | 112/140 | 135/140 | 91/102 | 98/102 |
| verbal-report | 1 | band | Jacobian lens | 0/140 | 1/140 | 0/102 | 1/102 |
| verbal-report | 1 | band | unembedding | 37/140 | 90/140 | 29/102 | 66/102 |
| verbal-report | 2 | 0-32 | Jacobian lens | 104/140 | 122/140 | 87/102 | 97/102 |
| verbal-report | 2 | 0-32 | unembedding | 132/140 | 139/140 | 100/102 | 102/102 |
| verbal-report | 2 | band | Jacobian lens | 16/140 | 27/140 | 16/102 | 27/102 |
| verbal-report | 2 | band | unembedding | 80/140 | 109/140 | 71/102 | 86/102 |

Full cells (all selection x edit combinations, KL, edit norms): `summary.json`.
