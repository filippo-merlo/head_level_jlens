# Head-level Jacobian lens -- google/gemma-3-4b-it

## The instrument
- `J^z` fitted at `o_proj`'s input on 1000 wikitext prompts as 4 disjoint shards, all 34 layers. sha256 `5c9f6f1263f4`.
- `W_O` gate at L33: row cosine 0.990 against `diag(g_post) W_O`. `W_O` is 2560x2048, so there is no identity arm: a hook slipped to `o_proj`'s *output* would read ~0 here instead.

## 1. A head's write is legible through the transport
*Best site: minimum rank over the 8 heads of a layer, then over the region's layers, for head arms and the whole-layer reference alike; each rate a mean over items. The paired J-head minus Logit-head difference is primary -- both arms read the same activations on the same items.*

Heads write a median **0.0135** of the residual stream they join (max 0.0970), with median ablation KL **0.03048 nats** (max 0.2988), over 240 items at each one's readout position. Every number below describes the *direction* of a write that size.

Items scored, where the candidate rule left an item with no scoreable intermediate under this tokenizer (the rest of the set is unaffected): multihop 84/93 (nhop-alphabet-element: 26, nhop-fortnight-element: 14, nhop-soccer-element: 11, nhop-chess-element: 16, nhop-suit-element: 13, nhop-clock-element: 12, nhop-deck-element: 52, roman-states-us: 50, half-clock-hours: 12).

**pass@1**, all layers

| set | J-head | Logit-head | J-layer | ratio | paired diff |
|---|---|---|---|---|---|
| multihop | 0.548 | 0.565 | 0.345 | 0.97x | -0.018 |
| multilingual | 0.694 | 0.745 | 0.227 | 0.93x | -0.051 |
| poetry | 0.204 | 0.194 | 0.010 | 1.05x | +0.010 |
| order-ops | 0.455 | 0.500 | 0.309 | 0.91x | -0.045 |
| association | 0.216 | 0.176 | 0.010 | 1.22x | +0.039 |
| typo | 0.958 | 0.969 | 0.312 | 0.99x | -0.010 |

Range 0.91x-1.22x over the logit lens on the same vector; the paired difference is positive on 2/6 sets.

**pass@5**, all layers

| set | J-head | Logit-head | J-layer | ratio | paired diff |
|---|---|---|---|---|---|
| multihop | 0.738 | 0.812 | 0.571 | 0.91x | -0.073 |
| multilingual | 0.799 | 0.815 | 0.320 | 0.98x | -0.016 |
| poetry | 0.378 | 0.418 | 0.031 | 0.90x | -0.041 |
| order-ops | 0.773 | 0.845 | 0.500 | 0.91x | -0.073 |
| association | 0.382 | 0.363 | 0.069 | 1.05x | +0.020 |
| typo | 0.990 | 0.990 | 0.490 | 1.00x | +0.000 |

Range 0.90x-1.05x over the logit lens on the same vector; the paired difference is positive on 1/6 sets.

**pass@25**, all layers

| set | J-head | Logit-head | J-layer | ratio | paired diff |
|---|---|---|---|---|---|
| multihop | 0.972 | 0.972 | 0.813 | 1.00x | +0.000 |
| multilingual | 0.864 | 0.886 | 0.467 | 0.98x | -0.021 |
| poetry | 0.663 | 0.755 | 0.061 | 0.88x | -0.092 |
| order-ops | 0.936 | 0.964 | 0.755 | 0.97x | -0.027 |
| association | 0.588 | 0.608 | 0.127 | 0.97x | -0.020 |
| typo | 1.000 | 1.000 | 0.781 | 1.00x | +0.000 |

Range 0.88x-1.00x over the logit lens on the same vector; the paired difference is positive on 0/6 sets.

**Sites within the top 1**, of 272: mean count per intermediate on its own prompt, and on a control prompt from the same set not annotated with it

| set | J-head | J-head control | Logit-head | Logit-head control | paired diff of excess |
|---|---|---|---|---|---|
| multihop | 1.2 | 0.0 | 1.1 | 0.0 | +0.0 |
| multilingual | 1.8 | 0.0 | 1.9 | 0.1 | -0.1 |
| poetry | 0.3 | 0.0 | 0.3 | 0.0 | +0.0 |
| order-ops | 0.9 | 0.4 | 1.0 | 0.4 | -0.1 |
| association | 0.4 | 0.0 | 0.3 | 0.0 | +0.1 |
| typo | 4.4 | 0.0 | 4.4 | 0.0 | +0.1 |

**Sites within the top 5**, of 272: mean count per intermediate on its own prompt, and on a control prompt from the same set not annotated with it

| set | J-head | J-head control | Logit-head | Logit-head control | paired diff of excess |
|---|---|---|---|---|---|
| multihop | 3.2 | 0.2 | 2.7 | 0.2 | +0.6 |
| multilingual | 4.0 | 0.2 | 3.9 | 0.2 | +0.1 |
| poetry | 0.6 | 0.1 | 0.7 | 0.2 | +0.0 |
| order-ops | 2.4 | 1.4 | 3.5 | 2.1 | -0.3 |
| association | 1.0 | 0.1 | 0.8 | 0.1 | +0.2 |
| typo | 7.7 | 0.1 | 7.8 | 0.1 | -0.0 |

**Sites within the top 25**, of 272: mean count per intermediate on its own prompt, and on a control prompt from the same set not annotated with it

| set | J-head | J-head control | Logit-head | Logit-head control | paired diff of excess |
|---|---|---|---|---|---|
| multihop | 7.1 | 0.7 | 6.6 | 0.9 | +0.8 |
| multilingual | 7.4 | 0.6 | 7.3 | 0.7 | +0.2 |
| poetry | 1.5 | 0.3 | 2.0 | 0.5 | -0.3 |
| order-ops | 5.8 | 3.7 | 9.4 | 7.1 | -0.2 |
| association | 2.4 | 0.3 | 2.0 | 0.4 | +0.5 |
| typo | 12.7 | 0.5 | 12.7 | 0.5 | +0.1 |

**Per layer**, pass@5, J-head over Logit-head

| set | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multihop | -- | -- | 0.00 | -- | -- | 0.00 | -- | 0.00 | 4.00 | -- | 0.71 | 5.00 | -- | -- | 2.00 | 1.71 |
| multilingual | -- | 0.00 | -- | 0.00 | -- | 0.00 | 0.26 | 0.00 | 4.12 | 1.00 | 2.25 | 2.36 | 4.00 | -- | 2.00 | 1.78 |
| poetry | -- | -- | -- | -- | -- | -- | -- | -- | 1.00 | -- | -- | -- | -- | -- | -- | 2.00 |
| order-ops | -- | -- | -- | -- | -- | 2.00 | 1.00 | 0.46 | 7.00 | 0.20 | 1.00 | 0.25 | -- | 0.00 | -- | 0.38 |
| association | -- | -- | -- | 0.00 | -- | 0.00 | 0.00 | -- | -- | -- | 0.50 | 7.00 | -- | -- | -- | 3.00 |
| typo | -- | -- | -- | -- | -- | 0.00 | 0.17 | 1.50 | 2.40 | -- | 4.00 | 2.00 | 1.00 | 1.00 | 1.00 | 0.85 |

| set | L16 | L17 | L18 | L19 | L20 | L21 | L22 | L23 | L24 | L25 | L26 | L27 | L28 | L29 | L30 | L31 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multihop | 0.92 | 1.86 | -- | 2.38 | 0.89 | 0.70 | 1.59 | 1.55 | 1.08 | 0.93 | 0.97 | 2.00 | 0.95 | 1.00 | 1.17 | 1.33 |
| multilingual | 1.00 | 1.46 | 0.30 | 1.09 | 0.89 | 0.83 | 1.01 | 0.99 | 0.92 | 0.64 | 0.90 | 0.93 | 1.08 | 1.00 | 1.02 | 1.04 |
| poetry | 1.00 | 0.33 | 0.00 | -- | 0.50 | 0.33 | -- | -- | 0.50 | 1.00 | 1.00 | 0.43 | 1.20 | 0.00 | 1.33 | 0.00 |
| order-ops | 0.71 | 0.23 | 1.92 | 2.00 | -- | 3.40 | 7.00 | 1.00 | 0.48 | 0.50 | 0.93 | 0.95 | 1.06 | 0.72 | 1.33 | 0.44 |
| association | 1.00 | 1.40 | -- | 1.25 | 1.00 | -- | 1.12 | -- | 0.86 | 0.67 | 0.78 | 1.00 | 1.17 | -- | 2.00 | 1.00 |
| typo | 0.88 | 0.93 | 1.30 | 0.59 | 1.64 | 0.71 | 1.00 | 1.07 | 1.00 | 0.94 | 0.94 | 0.88 | 0.77 | 1.01 | 1.00 | 0.75 |

| set | L32 | L33 |
|---|---|---|
| multihop | 0.77 | 1.11 |
| multilingual | 1.06 | 1.06 |
| poetry | -- | -- |
| order-ops | 0.63 | 0.00 |
| association | -- | 1.00 |
| typo | 1.09 | 1.08 |


**Best-site agreement**, J-head and Logit-head: fraction of (item, intermediate) pairs whose best site is the same under both readouts

| set | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multihop | 0.23 | 0.11 | 0.04 | 0.22 | 0.29 | 0.18 | 0.21 | 0.06 | 0.17 | 0.13 | 0.20 | 0.30 | 0.20 | 0.21 | 0.23 | 0.41 |
| multilingual | 0.13 | 0.16 | 0.16 | 0.19 | 0.17 | 0.11 | 0.31 | 0.08 | 0.27 | 0.13 | 0.12 | 0.37 | 0.24 | 0.16 | 0.30 | 0.48 |
| poetry | 0.12 | 0.09 | 0.19 | 0.18 | 0.28 | 0.07 | 0.11 | 0.10 | 0.13 | 0.22 | 0.07 | 0.21 | 0.17 | 0.17 | 0.16 | 0.31 |
| order-ops | 0.30 | 0.20 | 0.05 | 0.39 | 0.29 | 0.05 | 0.07 | 0.07 | 0.25 | 0.06 | 0.06 | 0.29 | 0.17 | 0.13 | 0.12 | 0.32 |
| association | 0.19 | 0.17 | 0.16 | 0.18 | 0.27 | 0.14 | 0.28 | 0.11 | 0.18 | 0.19 | 0.16 | 0.37 | 0.31 | 0.27 | 0.23 | 0.49 |
| typo | 0.15 | 0.10 | 0.19 | 0.26 | 0.15 | 0.19 | 0.24 | 0.12 | 0.40 | 0.23 | 0.15 | 0.48 | 0.29 | 0.18 | 0.46 | 0.50 |

| set | L16 | L17 | L18 | L19 | L20 | L21 | L22 | L23 | L24 | L25 | L26 | L27 | L28 | L29 | L30 | L31 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multihop | 0.50 | 0.45 | 0.31 | 0.50 | 0.56 | 0.66 | 0.54 | 0.68 | 0.80 | 0.67 | 0.79 | 0.67 | 0.82 | 0.67 | 0.76 | 0.76 |
| multilingual | 0.58 | 0.56 | 0.40 | 0.52 | 0.64 | 0.56 | 0.60 | 0.68 | 0.81 | 0.70 | 0.65 | 0.64 | 0.68 | 0.75 | 0.75 | 0.66 |
| poetry | 0.48 | 0.36 | 0.41 | 0.45 | 0.56 | 0.53 | 0.51 | 0.46 | 0.44 | 0.65 | 0.78 | 0.78 | 0.47 | 0.46 | 0.67 | 0.65 |
| order-ops | 0.40 | 0.43 | 0.53 | 0.40 | 0.28 | 0.41 | 0.47 | 0.64 | 0.49 | 0.52 | 0.55 | 0.57 | 0.58 | 0.82 | 0.70 | 0.58 |
| association | 0.40 | 0.55 | 0.18 | 0.62 | 0.59 | 0.26 | 0.73 | 0.47 | 0.75 | 0.43 | 0.53 | 0.57 | 0.71 | 0.59 | 0.77 | 0.59 |
| typo | 0.52 | 0.64 | 0.57 | 0.61 | 0.69 | 0.85 | 0.60 | 0.80 | 0.83 | 0.82 | 0.96 | 0.79 | 0.86 | 0.99 | 0.86 | 0.70 |

| set | L32 | L33 |
|---|---|---|
| multihop | 0.73 | 0.88 |
| multilingual | 0.54 | 0.88 |
| poetry | 0.41 | 0.80 |
| order-ops | 0.54 | 0.48 |
| association | 0.52 | 0.83 |
| typo | 0.82 | 0.91 |

| set | all layers |
|---|---|
| multihop | 0.69 |
| multilingual | 0.84 |
| poetry | 0.59 |
| order-ops | 0.55 |
| association | 0.34 |
| typo | 0.97 |

## 2-3. The coordinate is load-bearing, and how many sites carry it
*One independent variable: which sites are edited. Same trials, same intervention, same arms throughout. At each site count three rows edit the same number of sites and differ only in which: the highest-loading on the source concept (`top_k`), a random draw (`random_k`), the lowest-loading (`bottom_k`); `active` is the per-site-null threshold set, whose count is an outcome, with its two controls drawn at that count.*

**verbal-report** -- 140 trials

The greedy answer is a member of its category on 0 of 140 trials; elsewhere it is '\n\n'. Where it is not, the direction swapped out is not a concept direction and success means the target overtakes that token.

`layer_wide`, the paper's own intervention -- the Jacobian lens on the residual stream at every block 0-32 -- is the reference the head-level rows are read against: head_jlens a1 0.564; head_jlens a2 0.750; matched_norm a1 0.000; matched_norm a2 0.000. It is block-anchored through the Jacobian lens, not `zin` through `J^z`: a second patch family, marked rather than blended in. Without it a null at any count would say nothing about the count and everything about the setup.

`head_jlens` arm, swap rate by site count

| sites | top a1 | random a1 | bottom a1 | top a2 | random a2 | bottom a2 |
|---|---|---|---|---|---|---|
| 1 | 0.000 | 0.000 | 0.000 | 0.007 | 0.000 | 0.000 |
| 8 | 0.050 | 0.014 | 0.000 | 0.079 | 0.014 | 0.000 |
| 16 | 0.064 | 0.007 | 0.000 | 0.229 | 0.021 | 0.000 |
| 32 | 0.100 | 0.007 | 0.000 | 0.407 | 0.014 | 0.000 |
| 64 | 0.321 | 0.000 | 0.000 | 0.779 | 0.050 | 0.000 |
| active (median 21.5) | 0.093 | 0.000 | 0.000 | 0.307 | 0.014 | 0.000 |

`matched_norm` arm, swap rate by site count

| sites | top a1 | random a1 | bottom a1 | top a2 | random a2 | bottom a2 |
|---|---|---|---|---|---|---|
| 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 64 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| active (median 21.5) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Selection descriptives. A concept is active at 21.8 of 272 sites per trial (median 21.5, range 19-25, 0 trials with none), against a per-site 99th-percentile null. Per count, top / random / bottom: mean loading of the selected sites, their mean relative write norm (from read.py), and the total edit norm the `head_jlens` edit realised at alpha=1. The controls are matched in count, not in size -- high loading implies a large `|c_s|` implies a large edit -- so the last column is the size difference left visible rather than equalised away.

| sites | mean loading | relative write norm | total edit norm |
|---|---|---|---|
| 1 | 0.260 / -0.004 / -0.511 | 0.020 / 0.017 / 0.015 | 5.2 / 7.6 / 16.1 |
| 8 | 0.216 / -0.008 / -0.311 | 0.023 / 0.018 / 0.014 | 28.6 / 62.9 / 109.9 |
| 16 | 0.179 / -0.007 / -0.244 | 0.021 / 0.017 / 0.014 | 65.6 / 95.2 / 158.5 |
| 32 | 0.147 / -0.006 / -0.183 | 0.020 / 0.018 / 0.014 | 117.5 / 145.3 / 208.2 |
| 64 | 0.110 / -0.005 / -0.130 | 0.020 / 0.018 / 0.016 | 200.5 / 215.1 / 228.6 |
| active (median 21.5) | 0.153 / -0.008 / -0.216 | 0.017 / 0.017 / 0.015 | 39.4 / 115.9 / 204.0 |

The concept-selected sets span (distinct layers / distinct (layer, GQA-group) pairs): `top_1` 1.0/1.0, `top_8` 6.9/8.0, `top_16` 13.5/15.9, `top_32` 24.0/30.6, `top_64` 32.1/55.6, `active` 16.9/21.0 -- a set that kept picking group-mates would not be that many independent sites, since one `W_V` is shared within a GQA group.

**probe-swap** -- 46 trials

`layer_wide`, the paper's own intervention -- the Jacobian lens on the residual stream at every block 0-32 -- is the reference the head-level rows are read against: head_jlens a1 0.478; head_jlens a2 0.174; matched_norm a1 0.087; matched_norm a2 0.043. It is block-anchored through the Jacobian lens, not `zin` through `J^z`: a second patch family, marked rather than blended in. Without it a null at any count would say nothing about the count and everything about the setup.

`head_jlens` arm, swap rate by site count

| sites | top a1 | random a1 | bottom a1 | top a2 | random a2 | bottom a2 |
|---|---|---|---|---|---|---|
| 1 | 0.304 | 0.043 | 0.022 | 0.370 | 0.065 | 0.022 |
| 8 | 0.457 | 0.065 | 0.022 | 0.543 | 0.065 | 0.022 |
| 16 | 0.522 | 0.065 | 0.022 | 0.500 | 0.065 | 0.000 |
| 32 | 0.565 | 0.087 | 0.022 | 0.500 | 0.152 | 0.000 |
| 64 | 0.565 | 0.152 | 0.000 | 0.522 | 0.217 | 0.000 |
| active (median 46.0) | 0.565 | 0.152 | 0.022 | 0.500 | 0.217 | 0.000 |

`matched_norm` arm, swap rate by site count

| sites | top a1 | random a1 | bottom a1 | top a2 | random a2 | bottom a2 |
|---|---|---|---|---|---|---|
| 1 | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 |
| 8 | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 |
| 16 | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 |
| 32 | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 | 0.043 |
| 64 | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 | 0.043 |
| active (median 46.0) | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 | 0.043 |

Selection descriptives. A concept is active at 46.0 of 272 sites per trial (median 46.0, range 16-74, 0 trials with none), against a per-site 99th-percentile null. Per count, top / random / bottom: mean loading of the selected sites, their mean relative write norm (from read.py), and the total edit norm the `head_jlens` edit realised at alpha=1. The controls are matched in count, not in size -- high loading implies a large `|c_s|` implies a large edit -- so the last column is the size difference left visible rather than equalised away.

| sites | mean loading | relative write norm | total edit norm |
|---|---|---|---|
| 1 | 0.586 / 0.012 / -0.446 | 0.014 / 0.016 / 0.022 | 16.6 / 5.9 / 9.2 |
| 8 | 0.411 / 0.020 / -0.332 | 0.016 / 0.017 / 0.020 | 26.8 / 29.3 / 28.7 |
| 16 | 0.350 / 0.024 / -0.274 | 0.016 / 0.017 / 0.019 | 29.0 / 52.1 / 45.7 |
| 32 | 0.279 / 0.025 / -0.208 | 0.016 / 0.017 / 0.019 | 36.7 / 72.5 / 83.6 |
| 64 | 0.205 / 0.029 / -0.143 | 0.016 / 0.017 / 0.019 | 80.3 / 119.6 / 139.3 |
| active (median 46.0) | 0.233 / 0.027 / -0.172 | 0.016 / 0.017 / 0.019 | 48.1 / 97.2 / 108.7 |

The concept-selected sets span (distinct layers / distinct (layer, GQA-group) pairs): `top_1` 1.0/1.0, `top_8` 7.4/7.9, `top_16` 12.9/15.5, `top_32` 20.0/29.7, `top_64` 27.3/56.8, `active` 23.1/41.9 -- a set that kept picking group-mates would not be that many independent sites, since one `W_V` is shared within a GQA group.

alpha=1 exchanges the two lens coordinates; alpha=2 extrapolates past the exchange (the v_s coordinate becomes 2c_1 - c_0). They are different operations, not two strengths of one. Read both, never their maximum.

## What this does not establish
- **Legibility is not magnitude.** RMSNorm is scale-invariant, so a head contributing almost nothing still yields a confident-looking top-1. The heads above write a median 1.4% of the residual stream they join.
- **probe-swap deviates.** Its items substitute lens directions for the shipped linear-probe directions, and they are not the paper's headline 50-prompt set, so those rates are not comparable to the paper's.
