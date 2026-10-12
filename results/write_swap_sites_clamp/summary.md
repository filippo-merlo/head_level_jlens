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

probe-swap: median active count J 79, Logit 37; mean J/Logit top-set overlap top_1: 0.6, top_8: 4.3, top_16: 8.4, top_32: 15.2, top_64: 25.5, active: 23.6
verbal-report: median active count J 50, Logit 32; mean J/Logit top-set overlap top_1: 0.3, top_8: 4.0, top_16: 7.4, top_32: 12.9, top_64: 23.9, active: 16.7

## probe-swap, alpha = 1 (45 trials)

| sites | J sel+edit | L sel+edit | end-to-end diff | random sites J / L | bottom J / L | random direction J / L | direction diff at J sites | selection diff, J edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | +0.000 | 0 / 0 | 0 / 0 | 0 / 0 | +0.000 | +0.000 |
| 8 | 16 | 15 | +0.022 | 0 / 0 | 0 / 0 | 0 / 0 | +0.133 | +0.000 |
| 16 | 20 | 16 | +0.089 | 0 / 0 | 0 / 0 | 0 / 0 | +0.089 | +0.022 |
| 32 | 22 | 16 | +0.133 | 0 / 0 | 0 / 0 | 0 / 0 | +0.133 | +0.044 |
| 64 | 23 | 16 | +0.156 | 0 / 0 | 0 / 0 | 0 / 0 | +0.133 | +0.044 |
| active | 23 | 16 | +0.156 | 0 / 0 | 0 / 0 | 0 / 0 | +0.133 | +0.067 |
| layer-wide | 20 | 8 | +0.267 | | | 2 / 2 | | |
| layer-wide, band 12-28 | 21 | 7 | | | | 1 / 1 | | |

Mean KL from clean, top_64 J / L: 3.145 / 1.860; layer-wide J / L: 3.402 / 0.874; band J / L: 3.186 / 0.705.
Mean total edit norm at each lens's own top sites, J / L: 1: 2.8 / 3.2, 8: 5.0 / 5.4, 16: 5.7 / 6.5, 32: 6.6 / 7.7, 64: 7.4 / 9.2.

## probe-swap, alpha = 2 (45 trials)

| sites | J sel+edit | L sel+edit | end-to-end diff | random sites J / L | bottom J / L | random direction J / L | direction diff at J sites | selection diff, J edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 7 | +0.022 | 0 / 0 | 0 / 0 | 0 / 0 | +0.022 | +0.000 |
| 8 | 23 | 18 | +0.111 | 0 / 0 | 0 / 0 | 0 / 0 | +0.089 | +0.044 |
| 16 | 23 | 19 | +0.089 | 0 / 0 | 0 / 0 | 0 / 0 | +0.089 | +0.044 |
| 32 | 25 | 21 | +0.089 | 0 / 0 | 1 / 0 | 0 / 0 | +0.089 | +0.067 |
| 64 | 25 | 24 | +0.022 | 2 / 1 | 1 / 0 | 0 / 0 | +0.044 | +0.044 |
| active | 23 | 20 | +0.067 | 3 / 2 | 1 / 0 | 0 / 0 | +0.044 | +0.000 |
| layer-wide | 10 | 14 | -0.089 | | | 2 / 1 | | |
| layer-wide, band 12-28 | 15 | 16 | | | | 1 / 1 | | |

Mean KL from clean, top_64 J / L: 6.091 / 4.864; layer-wide J / L: 7.644 / 4.479; band J / L: 6.931 / 3.512.
Mean total edit norm at each lens's own top sites, J / L: 1: 5.5 / 6.5, 8: 9.6 / 10.5, 16: 11.2 / 12.7, 32: 13.0 / 15.2, 64: 14.8 / 18.3.

## verbal-report, alpha = 1 (140 trials)

| sites | J sel+edit | L sel+edit | end-to-end diff | random sites J / L | bottom J / L | random direction J / L | direction diff at J sites | selection diff, J edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | +0.007 | 0 / 0 | 0 / 0 | 0 / 0 | +0.007 | +0.007 |
| 8 | 18 | 8 | +0.071 | 0 / 0 | 0 / 0 | 0 / 0 | +0.086 | +0.029 |
| 16 | 32 | 20 | +0.086 | 0 / 0 | 0 / 0 | 0 / 0 | +0.114 | +0.036 |
| 32 | 45 | 34 | +0.079 | 0 / 0 | 1 / 0 | 0 / 0 | +0.129 | +0.043 |
| 64 | 57 | 51 | +0.043 | 0 / 0 | 0 / 0 | 0 / 0 | +0.179 | +0.029 |
| active | 62 | 35 | +0.193 | 0 / 0 | 0 / 0 | 0 / 0 | +0.186 | +0.136 |
| layer-wide | 81 | 64 | +0.121 | | | 1 / 1 | | |
| layer-wide, band 12-28 | 88 | 59 | | | | 1 / 0 | | |

Mean KL from clean, top_64 J / L: 0.414 / 0.206; layer-wide J / L: 1.352 / 0.413; band J / L: 0.908 / 0.287.
Mean total edit norm at each lens's own top sites, J / L: 1: 1.3 / 1.5, 8: 3.5 / 5.7, 16: 3.9 / 7.2, 32: 4.8 / 8.4, 64: 5.6 / 10.1.

## verbal-report, alpha = 2 (140 trials)

| sites | J sel+edit | L sel+edit | end-to-end diff | random sites J / L | bottom J / L | random direction J / L | direction diff at J sites | selection diff, J edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 3 | -0.007 | 0 / 0 | 0 / 0 | 0 / 0 | +0.000 | +0.000 |
| 8 | 48 | 35 | +0.093 | 0 / 0 | 0 / 0 | 0 / 0 | +0.071 | +0.064 |
| 16 | 63 | 59 | +0.029 | 0 / 0 | 0 / 0 | 0 / 0 | +0.136 | -0.029 |
| 32 | 92 | 82 | +0.071 | 0 / 0 | 1 / 0 | 1 / 1 | +0.164 | +0.021 |
| 64 | 107 | 109 | -0.014 | 4 / 1 | 0 / 0 | 1 / 1 | +0.186 | +0.043 |
| active | 106 | 83 | +0.164 | 2 / 0 | 0 / 0 | 1 / 0 | +0.143 | +0.114 |
| layer-wide | 103 | 110 | -0.050 | | | 2 / 2 | | |
| layer-wide, band 12-28 | 107 | 100 | | | | 1 / 0 | | |

Mean KL from clean, top_64 J / L: 1.888 / 0.843; layer-wide J / L: 4.378 / 1.652; band J / L: 3.020 / 1.172.
Mean total edit norm at each lens's own top sites, J / L: 1: 2.6 / 3.1, 8: 6.9 / 11.4, 16: 7.8 / 14.4, 32: 9.5 / 16.8, 64: 11.1 / 20.2.

## Layer-wide references, scored as Gurnee et al. score theirs

Target rank after the swap, top-1 and top-5, over all trials and over the trials whose
target was outside the clean top 10 (the paper swaps in only such verbal-report targets).
`band` is block outputs 12-28, the paper's workspace depth.

| task | alpha | scope | lens | top-1 | top-5 | top-1, outside top-10 | top-5, outside top-10 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| probe-swap | 1 | 0-30 | Jacobian lens | 20/45 | 39/45 | 17/36 | 31/36 |
| probe-swap | 1 | 0-30 | unembedding | 8/45 | 24/45 | 7/36 | 16/36 |
| probe-swap | 1 | band | Jacobian lens | 21/45 | 34/45 | 18/36 | 27/36 |
| probe-swap | 1 | band | unembedding | 7/45 | 20/45 | 6/36 | 12/36 |
| probe-swap | 2 | 0-30 | Jacobian lens | 10/45 | 33/45 | 8/36 | 28/36 |
| probe-swap | 2 | 0-30 | unembedding | 14/45 | 34/45 | 12/36 | 26/36 |
| probe-swap | 2 | band | Jacobian lens | 15/45 | 36/45 | 12/36 | 29/36 |
| probe-swap | 2 | band | unembedding | 16/45 | 33/45 | 14/36 | 25/36 |
| verbal-report | 1 | 0-30 | Jacobian lens | 81/140 | 125/140 | 51/82 | 74/82 |
| verbal-report | 1 | 0-30 | unembedding | 64/140 | 110/140 | 29/82 | 56/82 |
| verbal-report | 1 | band | Jacobian lens | 88/140 | 125/140 | 55/82 | 72/82 |
| verbal-report | 1 | band | unembedding | 59/140 | 107/140 | 28/82 | 54/82 |
| verbal-report | 2 | 0-30 | Jacobian lens | 103/140 | 129/140 | 64/82 | 77/82 |
| verbal-report | 2 | 0-30 | unembedding | 110/140 | 124/140 | 68/82 | 71/82 |
| verbal-report | 2 | band | Jacobian lens | 107/140 | 131/140 | 66/82 | 80/82 |
| verbal-report | 2 | band | unembedding | 100/140 | 122/140 | 56/82 | 67/82 |

Full cells (all selection x edit combinations, KL, edit norms): `summary.json`.
