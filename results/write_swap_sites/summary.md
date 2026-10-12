# Multi-site coordinate swap: J-head vs Logit-head

Experiment 2's sites, scope and swap (every position of every selected head, edited
pass), with the frame built through J^z (J) or W_O (Logit). The swap's size follows
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
| 8 | 10 | 11 | -0.022 | 0 / 0 | 0 / 0 | 0 / 0 | +0.089 | -0.044 |
| 16 | 17 | 11 | +0.133 | 0 / 0 | 0 / 0 | 0 / 0 | +0.111 | +0.089 |
| 32 | 18 | 12 | +0.133 | 0 / 0 | 0 / 0 | 0 / 0 | +0.111 | +0.067 |
| 64 | 19 | 12 | +0.156 | 0 / 0 | 0 / 0 | 0 / 0 | +0.089 | +0.089 |
| active | 17 | 13 | +0.089 | 0 / 0 | 0 / 0 | 0 / 0 | +0.067 | +0.022 |
| layer-wide | 8 | 0 | +0.178 | | | 2 / 1 | | |

Mean KL from clean, top_64 J / L: 0.950 / 0.662; layer-wide J / L: 0.646 / 0.150.
Mean total edit norm at each lens's own top sites, J / L: 1: 2.8 / 3.2, 8: 4.6 / 5.1, 16: 5.3 / 6.1, 32: 6.0 / 7.3, 64: 6.9 / 8.9.

## probe-swap, alpha = 2 (45 trials)

| sites | J sel+edit | L sel+edit | end-to-end diff | random sites J / L | bottom J / L | random direction J / L | direction diff at J sites | selection diff, J edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 7 | +0.022 | 0 / 0 | 0 / 0 | 0 / 0 | +0.022 | +0.000 |
| 8 | 17 | 14 | +0.067 | 0 / 0 | 0 / 0 | 0 / 0 | +0.067 | +0.044 |
| 16 | 16 | 13 | +0.067 | 0 / 0 | 0 / 0 | 0 / 0 | +0.067 | -0.022 |
| 32 | 14 | 12 | +0.044 | 0 / 0 | 2 / 0 | 0 / 0 | +0.000 | +0.067 |
| 64 | 12 | 12 | +0.000 | 1 / 1 | 2 / 0 | 0 / 0 | -0.044 | -0.067 |
| active | 12 | 11 | +0.022 | 2 / 1 | 3 / 0 | 0 / 0 | +0.000 | -0.022 |
| layer-wide | 0 | 0 | +0.000 | | | 1 / 1 | | |

Mean KL from clean, top_64 J / L: 0.874 / 0.852; layer-wide J / L: 67.113 / 73.739.
Mean total edit norm at each lens's own top sites, J / L: 1: 5.5 / 6.5, 8: 8.4 / 9.4, 16: 10.0 / 11.5, 32: 11.4 / 14.1, 64: 13.0 / 17.3.

## verbal-report, alpha = 1 (140 trials)

| sites | J sel+edit | L sel+edit | end-to-end diff | random sites J / L | bottom J / L | random direction J / L | direction diff at J sites | selection diff, J edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | +0.007 | 0 / 0 | 0 / 0 | 0 / 0 | +0.007 | +0.007 |
| 8 | 17 | 7 | +0.071 | 0 / 0 | 0 / 0 | 0 / 0 | +0.079 | +0.043 |
| 16 | 28 | 17 | +0.079 | 0 / 0 | 0 / 0 | 0 / 0 | +0.107 | +0.036 |
| 32 | 34 | 30 | +0.029 | 0 / 0 | 1 / 0 | 0 / 0 | +0.100 | +0.000 |
| 64 | 41 | 42 | -0.007 | 0 / 0 | 0 / 0 | 0 / 0 | +0.114 | +0.000 |
| active | 49 | 30 | +0.136 | 0 / 0 | 0 / 0 | 0 / 0 | +0.143 | +0.114 |
| layer-wide | 41 | 36 | +0.036 | | | 1 / 1 | | |

Mean KL from clean, top_64 J / L: 0.313 / 0.174; layer-wide J / L: 0.529 / 0.157.
Mean total edit norm at each lens's own top sites, J / L: 1: 1.3 / 1.5, 8: 3.3 / 5.7, 16: 3.7 / 7.1, 32: 4.5 / 8.3, 64: 5.3 / 10.1.

## verbal-report, alpha = 2 (140 trials)

| sites | J sel+edit | L sel+edit | end-to-end diff | random sites J / L | bottom J / L | random direction J / L | direction diff at J sites | selection diff, J edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 3 | +0.000 | 0 / 0 | 0 / 0 | 0 / 0 | +0.007 | +0.007 |
| 8 | 40 | 32 | +0.057 | 0 / 0 | 0 / 0 | 0 / 0 | +0.057 | +0.029 |
| 16 | 45 | 43 | +0.014 | 0 / 0 | 1 / 0 | 0 / 1 | +0.064 | +0.029 |
| 32 | 49 | 60 | -0.079 | 0 / 0 | 3 / 0 | 1 / 1 | +0.043 | -0.050 |
| 64 | 55 | 85 | -0.214 | 2 / 2 | 2 / 0 | 1 / 1 | +0.007 | -0.050 |
| active | 57 | 57 | +0.000 | 1 / 1 | 2 / 0 | 0 / 0 | -0.043 | +0.021 |
| layer-wide | 74 | 48 | +0.186 | | | 2 / 2 | | |

Mean KL from clean, top_64 J / L: 0.989 / 0.495; layer-wide J / L: 74.166 / 79.308.
Mean total edit norm at each lens's own top sites, J / L: 1: 2.6 / 3.1, 8: 6.3 / 11.2, 16: 7.1 / 14.1, 32: 8.5 / 16.6, 64: 10.3 / 20.2.

Full cells (all selection x edit combinations, KL, edit norms): `summary.json`.
