# Head-level Jacobian lens -- Qwen/Qwen3.5-4B

## The instrument
- `J^z` fitted at `o_proj`'s input on 1000 wikitext prompts as 4 disjoint shards, the 8 attention layers of 32 (3, 7, ..., 31). sha256 `249929f24139`.
- `W_O` gate at L31: row cosine 0.904 against `W_O`. `W_O` is 2560x4096, so there is no identity arm: a hook slipped to `o_proj`'s *output* would read ~0 here instead.

## 1. A head's write is legible through the transport
*Best site: minimum rank over the 16 heads of a layer, then over the region's layers, for head arms and the whole-layer reference alike; each rate a mean over items. The paired J-head minus Logit-head difference is primary -- both arms read the same activations on the same items.*

Heads write a median **0.0694** of the residual stream they join (max 0.1775), with median ablation KL **0.00441 nats** (max 0.0374), over 240 items at each one's readout position. Every number below describes the *direction* of a write that size.

Items scored, where the candidate rule left an item with no scoreable intermediate under this tokenizer (the rest of the set is unaffected): multihop 84/93 (nhop-alphabet-element: 26, nhop-fortnight-element: 14, nhop-soccer-element: 11, nhop-chess-element: 16, nhop-suit-element: 13, nhop-clock-element: 12, nhop-deck-element: 52, roman-states-us: 50, half-clock-hours: 12).

**pass@1**, all layers

| set | J-head | Logit-head | J-layer | ratio | paired diff |
|---|---|---|---|---|---|
| multihop | 0.179 | 0.113 | 0.161 | 1.58x | +0.065 |
| multilingual | 0.397 | 0.290 | 0.178 | 1.37x | +0.107 |
| poetry | 0.010 | 0.010 | 0.000 | 1.00x | +0.000 |
| order-ops | 0.655 | 0.545 | 0.173 | 1.20x | +0.109 |
| association | 0.088 | 0.020 | 0.000 | 4.50x | +0.069 |
| typo | 0.938 | 0.781 | 0.208 | 1.20x | +0.156 |

Range 1.00x-4.50x over the logit lens on the same vector; the paired difference is positive on 5/6 sets.

**pass@5**, all layers

| set | J-head | Logit-head | J-layer | ratio | paired diff |
|---|---|---|---|---|---|
| multihop | 0.444 | 0.200 | 0.387 | 2.22x | +0.244 |
| multilingual | 0.598 | 0.486 | 0.292 | 1.23x | +0.112 |
| poetry | 0.041 | 0.041 | 0.000 | 1.00x | +0.000 |
| order-ops | 0.818 | 0.800 | 0.345 | 1.02x | +0.018 |
| association | 0.186 | 0.039 | 0.000 | 4.75x | +0.147 |
| typo | 1.000 | 0.927 | 0.323 | 1.08x | +0.073 |

Range 1.00x-4.75x over the logit lens on the same vector; the paired difference is positive on 5/6 sets.

**pass@25**, all layers

| set | J-head | Logit-head | J-layer | ratio | paired diff |
|---|---|---|---|---|---|
| multihop | 0.784 | 0.462 | 0.587 | 1.70x | +0.321 |
| multilingual | 0.785 | 0.643 | 0.425 | 1.22x | +0.143 |
| poetry | 0.153 | 0.112 | 0.010 | 1.36x | +0.041 |
| order-ops | 0.964 | 0.864 | 0.691 | 1.12x | +0.100 |
| association | 0.382 | 0.186 | 0.020 | 2.05x | +0.196 |
| typo | 1.000 | 1.000 | 0.573 | 1.00x | +0.000 |

Range 1.00x-2.05x over the logit lens on the same vector; the paired difference is positive on 5/6 sets.

**Sites within the top 1**, of 128: mean count per intermediate on its own prompt, and on a control prompt from the same set not annotated with it

| set | J-head | J-head control | Logit-head | Logit-head control | paired diff of excess |
|---|---|---|---|---|---|
| multihop | 0.3 | 0.0 | 0.2 | 0.0 | +0.1 |
| multilingual | 0.8 | 0.0 | 0.5 | 0.0 | +0.3 |
| poetry | 0.0 | 0.0 | 0.0 | 0.0 | -0.0 |
| order-ops | 1.6 | 0.6 | 0.9 | 0.4 | +0.4 |
| association | 0.2 | 0.0 | 0.0 | 0.0 | +0.1 |
| typo | 2.3 | 0.0 | 1.5 | 0.0 | +0.8 |

**Sites within the top 5**, of 128: mean count per intermediate on its own prompt, and on a control prompt from the same set not annotated with it

| set | J-head | J-head control | Logit-head | Logit-head control | paired diff of excess |
|---|---|---|---|---|---|
| multihop | 0.7 | 0.1 | 0.4 | 0.1 | +0.3 |
| multilingual | 1.9 | 0.1 | 1.0 | 0.0 | +0.8 |
| poetry | 0.0 | 0.1 | 0.0 | 0.0 | -0.0 |
| order-ops | 3.4 | 2.1 | 2.7 | 1.5 | +0.1 |
| association | 0.4 | 0.0 | 0.0 | 0.0 | +0.4 |
| typo | 4.1 | 0.0 | 2.5 | 0.0 | +1.6 |

**Sites within the top 25**, of 128: mean count per intermediate on its own prompt, and on a control prompt from the same set not annotated with it

| set | J-head | J-head control | Logit-head | Logit-head control | paired diff of excess |
|---|---|---|---|---|---|
| multihop | 2.2 | 0.4 | 1.0 | 0.3 | +1.0 |
| multilingual | 3.7 | 0.3 | 1.8 | 0.1 | +1.8 |
| poetry | 0.2 | 0.1 | 0.1 | 0.1 | +0.0 |
| order-ops | 7.8 | 5.6 | 6.1 | 4.5 | +0.6 |
| association | 1.0 | 0.2 | 0.3 | 0.1 | +0.6 |
| typo | 6.9 | 0.1 | 3.8 | 0.1 | +3.2 |

**Per layer**, pass@5, J-head over Logit-head

| set | L3 | L7 | L11 | L15 | L19 | L23 | L27 | L31 |
|---|---|---|---|---|---|---|---|---|
| multihop | -- | -- | -- | -- | 3.67 | 2.14 | 1.25 | 1.60 |
| multilingual | 0.00 | 3.62 | 42.00 | -- | 3.50 | 1.11 | 1.31 | 1.35 |
| poetry | -- | -- | -- | -- | 3.00 | 0.00 | -- | 0.00 |
| order-ops | 5.25 | 0.50 | 10.00 | 6.00 | 3.06 | 1.48 | 0.92 | 0.83 |
| association | -- | -- | -- | -- | -- | 3.50 | 1.00 | 1.00 |
| typo | 0.00 | 6.50 | 4.67 | -- | 3.73 | 1.08 | 1.29 | 1.07 |


**Best-site agreement**, J-head and Logit-head: fraction of (item, intermediate) pairs whose best site is the same under both readouts

| set | L3 | L7 | L11 | L15 | L19 | L23 | L27 | L31 |
|---|---|---|---|---|---|---|---|---|
| multihop | 0.20 | 0.04 | 0.06 | 0.13 | 0.30 | 0.60 | 0.51 | 0.61 |
| multilingual | 0.11 | 0.14 | 0.11 | 0.13 | 0.42 | 0.69 | 0.61 | 0.64 |
| poetry | 0.04 | 0.10 | 0.11 | 0.17 | 0.22 | 0.57 | 0.36 | 0.36 |
| order-ops | 0.06 | 0.08 | 0.17 | 0.07 | 0.26 | 0.59 | 0.65 | 0.81 |
| association | 0.12 | 0.10 | 0.08 | 0.09 | 0.19 | 0.34 | 0.29 | 0.42 |
| typo | 0.16 | 0.14 | 0.19 | 0.18 | 0.40 | 0.91 | 0.84 | 0.79 |

| set | all layers |
|---|---|
| multihop | 0.40 |
| multilingual | 0.56 |
| poetry | 0.17 |
| order-ops | 0.73 |
| association | 0.14 |
| typo | 0.86 |

## 2-3. The coordinate is load-bearing, and how many sites carry it
*One independent variable: which sites are edited. Same trials, same intervention, same arms throughout. At each site count three rows edit the same number of sites and differ only in which: the highest-loading on the source concept (`top_k`), a random draw (`random_k`), the lowest-loading (`bottom_k`); `active` is the per-site-null threshold set, whose count is an outcome, with its two controls drawn at that count.*

**verbal-report** -- 140 trials

The greedy answer is a member of its category on 0 of 140 trials; elsewhere it is ' What', '\n', '\n\n'. Where it is not, the direction swapped out is not a concept direction and success means the target overtakes that token.

`layer_wide`, the paper's own intervention -- the Jacobian lens on the residual stream at every block 0-30 -- is the reference the head-level rows are read against: head_jlens a1 0.214; head_jlens a2 0.286; matched_norm a1 0.000; matched_norm a2 0.000. It is block-anchored through the Jacobian lens, not `zin` through `J^z`: a second patch family, marked rather than blended in. Without it a null at any count would say nothing about the count and everything about the setup.

`head_jlens` arm, swap rate by site count

| sites | top a1 | random a1 | bottom a1 | top a2 | random a2 | bottom a2 |
|---|---|---|---|---|---|---|
| 1 | 0.014 | 0.000 | 0.000 | 0.050 | 0.000 | 0.000 |
| 8 | 0.129 | 0.000 | 0.000 | 0.536 | 0.007 | 0.000 |
| 16 | 0.214 | 0.000 | 0.000 | 0.586 | 0.014 | 0.000 |
| 32 | 0.407 | 0.029 | 0.000 | 0.629 | 0.079 | 0.000 |
| 64 | 0.336 | 0.014 | 0.000 | 0.679 | 0.136 | 0.000 |
| active (median 16.0) | 0.514 | 0.000 | 0.000 | 0.643 | 0.021 | 0.000 |

`matched_norm` arm, swap rate by site count

| sites | top a1 | random a1 | bottom a1 | top a2 | random a2 | bottom a2 |
|---|---|---|---|---|---|---|
| 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 64 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| active (median 16.0) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Selection descriptives. A concept is active at 16.1 of 128 sites per trial (median 16.0, range 13-20, 0 trials with none), against a per-site 99th-percentile null. Per count, top / random / bottom: mean loading of the selected sites, their mean relative write norm (from read.py), and the total edit norm the `head_jlens` edit realised at alpha=1. The controls are matched in count, not in size -- high loading implies a large `|c_s|` implies a large edit -- so the last column is the size difference left visible rather than equalised away.

| sites | mean loading | relative write norm | total edit norm |
|---|---|---|---|
| 1 | 0.345 / 0.026 / -0.198 | 0.064 / 0.075 / 0.071 | 4.4 / 1.1 / 3.9 |
| 8 | 0.198 / 0.028 / -0.141 | 0.064 / 0.077 / 0.081 | 10.3 / 5.4 / 8.1 |
| 16 | 0.160 / 0.027 / -0.108 | 0.071 / 0.077 / 0.080 | 11.4 / 8.6 / 11.6 |
| 32 | 0.124 / 0.026 / -0.073 | 0.077 / 0.078 / 0.081 | 14.9 / 12.8 / 17.0 |
| 64 | 0.085 / 0.026 / -0.034 | 0.079 / 0.077 / 0.076 | 18.1 / 18.6 / 20.0 |
| active (median 16.0) | 0.141 / 0.027 / -0.108 | 0.071 / 0.077 / 0.080 | 18.5 / 8.5 / 10.6 |

The concept-selected sets span (distinct layers / distinct (layer, GQA-group) pairs): `top_1` 1.0/1.0, `top_8` 3.6/6.6, `top_16` 5.8/11.7, `top_32` 7.9/20.0, `top_64` 8.0/30.0, `active` 5.9/12.1 -- a set that kept picking group-mates would not be that many independent sites, since one `W_V` is shared within a GQA group.

**probe-swap** -- 53 trials

`layer_wide`, the paper's own intervention -- the Jacobian lens on the residual stream at every block 0-30 -- is the reference the head-level rows are read against: head_jlens a1 0.491; head_jlens a2 0.377; matched_norm a1 0.000; matched_norm a2 0.000. It is block-anchored through the Jacobian lens, not `zin` through `J^z`: a second patch family, marked rather than blended in. Without it a null at any count would say nothing about the count and everything about the setup.

`head_jlens` arm, swap rate by site count

| sites | top a1 | random a1 | bottom a1 | top a2 | random a2 | bottom a2 |
|---|---|---|---|---|---|---|
| 1 | 0.019 | 0.019 | 0.019 | 0.038 | 0.019 | 0.019 |
| 8 | 0.038 | 0.019 | 0.019 | 0.264 | 0.057 | 0.019 |
| 16 | 0.057 | 0.019 | 0.019 | 0.377 | 0.057 | 0.019 |
| 32 | 0.075 | 0.019 | 0.019 | 0.472 | 0.094 | 0.019 |
| 64 | 0.132 | 0.019 | 0.019 | 0.472 | 0.151 | 0.019 |
| active (median 17) | 0.094 | 0.019 | 0.019 | 0.358 | 0.057 | 0.019 |

`matched_norm` arm, swap rate by site count

| sites | top a1 | random a1 | bottom a1 | top a2 | random a2 | bottom a2 |
|---|---|---|---|---|---|---|
| 1 | 0.019 | 0.019 | 0.019 | 0.019 | 0.019 | 0.019 |
| 8 | 0.019 | 0.019 | 0.019 | 0.019 | 0.019 | 0.019 |
| 16 | 0.019 | 0.019 | 0.019 | 0.019 | 0.019 | 0.019 |
| 32 | 0.019 | 0.019 | 0.019 | 0.019 | 0.019 | 0.019 |
| 64 | 0.019 | 0.019 | 0.019 | 0.019 | 0.019 | 0.019 |
| active (median 17) | 0.019 | 0.019 | 0.019 | 0.019 | 0.019 | 0.019 |

Selection descriptives. A concept is active at 17.6 of 128 sites per trial (median 17, range 5-31, 0 trials with none), against a per-site 99th-percentile null. Per count, top / random / bottom: mean loading of the selected sites, their mean relative write norm (from read.py), and the total edit norm the `head_jlens` edit realised at alpha=1. The controls are matched in count, not in size -- high loading implies a large `|c_s|` implies a large edit -- so the last column is the size difference left visible rather than equalised away.

| sites | mean loading | relative write norm | total edit norm |
|---|---|---|---|
| 1 | 0.375 / 0.019 / -0.251 | 0.074 / 0.069 / 0.093 | 1.8 / 0.6 / 1.9 |
| 8 | 0.243 / 0.030 / -0.173 | 0.082 / 0.079 / 0.099 | 5.5 / 3.6 / 4.7 |
| 16 | 0.197 / 0.027 / -0.134 | 0.082 / 0.078 / 0.091 | 7.0 / 5.4 / 6.1 |
| 32 | 0.150 / 0.028 / -0.090 | 0.078 / 0.078 / 0.084 | 8.4 / 8.5 / 8.2 |
| 64 | 0.101 / 0.028 / -0.043 | 0.075 / 0.078 / 0.079 | 11.7 / 11.9 / 11.1 |
| active (median 17) | 0.178 / 0.028 / -0.127 | 0.082 / 0.079 / 0.091 | 7.5 / 5.7 / 6.3 |

The concept-selected sets span (distinct layers / distinct (layer, GQA-group) pairs): `top_1` 1.0/1.0, `top_8` 4.2/5.8, `top_16` 6.4/11.1, `top_32` 7.7/19.9, `top_64` 8.0/30.2, `active` 6.5/12.3 -- a set that kept picking group-mates would not be that many independent sites, since one `W_V` is shared within a GQA group.

alpha=1 exchanges the two lens coordinates; alpha=2 extrapolates past the exchange (the v_s coordinate becomes 2c_1 - c_0). They are different operations, not two strengths of one. Read both, never their maximum.

## What this does not establish
- **Legibility is not magnitude.** RMSNorm is scale-invariant, so a head contributing almost nothing still yields a confident-looking top-1. The heads above write a median 6.9% of the residual stream they join.
- **probe-swap deviates.** Its items substitute lens directions for the shipped linear-probe directions, and they are not the paper's headline 50-prompt set, so those rates are not comparable to the paper's.
