# Head-level Jacobian lens -- meta-llama/Llama-3.1-8B-Instruct

## The instrument
- `J^z` fitted at `o_proj`'s input on 1000 wikitext prompts as 4 disjoint shards, all 32 layers. sha256 `07a9a25aead3`.
- `W_O` gate at L31: row cosine 0.867 against `W_O` and -0.0002 against the identity. A hook slipped to `o_proj`'s *output* would swap these, which is what makes it a test rather than a trend.

## 1. A head's write is legible through the transport
*Best site: minimum rank over the 32 heads of a layer, then over the region's layers, for head arms and the whole-layer reference alike; each rate a mean over items. The paired J-head minus Logit-head difference is primary -- both arms read the same activations on the same items.*

Heads write a median **0.0393** of the residual stream they join (max 0.2990), with median ablation KL **0.00267 nats** (max 0.2986), over 240 items at each one's readout position. Every number below describes the *direction* of a write that size.

**pass@1**, all layers

| set | J-head | Logit-head | J-layer | ratio | paired diff |
|---|---|---|---|---|---|
| multihop | 0.649 | 0.505 | 0.452 | 1.28x | +0.143 |
| multilingual | 0.804 | 0.776 | 0.278 | 1.04x | +0.028 |
| poetry | 0.296 | 0.347 | 0.041 | 0.85x | -0.051 |
| order-ops | 0.764 | 0.682 | 0.409 | 1.12x | +0.082 |
| association | 0.324 | 0.294 | 0.049 | 1.10x | +0.029 |
| typo | 1.000 | 1.000 | 0.427 | 1.00x | +0.000 |

Range 0.85x-1.28x over the logit lens on the same vector; the paired difference is positive on 4/6 sets.

**pass@5**, all layers

| set | J-head | Logit-head | J-layer | ratio | paired diff |
|---|---|---|---|---|---|
| multihop | 0.916 | 0.799 | 0.735 | 1.15x | +0.116 |
| multilingual | 0.890 | 0.881 | 0.463 | 1.01x | +0.009 |
| poetry | 0.449 | 0.541 | 0.082 | 0.83x | -0.092 |
| order-ops | 0.982 | 0.882 | 0.655 | 1.11x | +0.100 |
| association | 0.627 | 0.559 | 0.147 | 1.12x | +0.069 |
| typo | 1.000 | 1.000 | 0.688 | 1.00x | +0.000 |

Range 0.83x-1.15x over the logit lens on the same vector; the paired difference is positive on 4/6 sets.

**pass@25**, all layers

| set | J-head | Logit-head | J-layer | ratio | paired diff |
|---|---|---|---|---|---|
| multihop | 0.989 | 0.946 | 0.842 | 1.05x | +0.043 |
| multilingual | 0.960 | 0.960 | 0.624 | 1.00x | +0.000 |
| poetry | 0.816 | 0.827 | 0.214 | 0.99x | -0.010 |
| order-ops | 1.000 | 0.991 | 0.909 | 1.01x | +0.009 |
| association | 0.843 | 0.863 | 0.284 | 0.98x | -0.020 |
| typo | 1.000 | 1.000 | 0.865 | 1.00x | +0.000 |

Range 0.98x-1.05x over the logit lens on the same vector; the paired difference is positive on 2/6 sets.

**Sites within the top 1**, of 1024: mean count per intermediate on its own prompt, and on a control prompt from the same set not annotated with it

| set | J-head | J-head control | Logit-head | Logit-head control | paired diff of excess |
|---|---|---|---|---|---|
| multihop | 1.8 | 0.1 | 1.3 | 0.1 | +0.5 |
| multilingual | 4.0 | 0.3 | 2.7 | 0.2 | +1.2 |
| poetry | 0.5 | 0.1 | 0.6 | 0.1 | -0.1 |
| order-ops | 3.2 | 2.1 | 2.1 | 0.9 | -0.1 |
| association | 1.0 | 0.1 | 0.6 | 0.1 | +0.4 |
| typo | 8.0 | 0.1 | 6.9 | 0.0 | +1.0 |

**Sites within the top 5**, of 1024: mean count per intermediate on its own prompt, and on a control prompt from the same set not annotated with it

| set | J-head | J-head control | Logit-head | Logit-head control | paired diff of excess |
|---|---|---|---|---|---|
| multihop | 5.4 | 0.5 | 3.0 | 0.2 | +2.1 |
| multilingual | 8.5 | 0.7 | 5.2 | 0.4 | +3.0 |
| poetry | 1.3 | 0.3 | 1.3 | 0.3 | +0.0 |
| order-ops | 10.9 | 7.5 | 4.9 | 3.0 | +1.4 |
| association | 2.8 | 0.3 | 1.4 | 0.2 | +1.3 |
| typo | 14.2 | 0.5 | 11.1 | 0.2 | +2.7 |

**Sites within the top 25**, of 1024: mean count per intermediate on its own prompt, and on a control prompt from the same set not annotated with it

| set | J-head | J-head control | Logit-head | Logit-head control | paired diff of excess |
|---|---|---|---|---|---|
| multihop | 12.7 | 2.0 | 5.7 | 0.9 | +5.9 |
| multilingual | 16.4 | 2.2 | 9.0 | 1.1 | +6.2 |
| poetry | 3.5 | 1.2 | 3.0 | 1.1 | +0.5 |
| order-ops | 34.0 | 29.0 | 12.2 | 9.5 | +2.2 |
| association | 6.8 | 1.0 | 3.2 | 0.7 | +3.4 |
| typo | 24.0 | 1.9 | 17.1 | 0.7 | +5.7 |

**Per layer**, pass@5, J-head over Logit-head

| set | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multihop | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | 1.38 | 3.97 |
| multilingual | 1.00 | 0.00 | -- | 3.00 | 8.00 | 1.00 | 6.00 | -- | 26.33 | 41.00 | -- | -- | 4.73 | 2.88 | 2.21 | 1.37 |
| poetry | -- | 1.00 | 0.00 | 0.00 | -- | -- | 1.00 | 0.00 | -- | -- | -- | 0.00 | -- | 1.00 | 1.00 | 1.00 |
| order-ops | -- | 12.50 | -- | -- | -- | 15.00 | -- | -- | -- | -- | -- | -- | -- | 14.00 | 20.00 | 7.00 |
| association | -- | -- | -- | -- | -- | 1.00 | -- | -- | 13.00 | 4.00 | -- | -- | 5.00 | 14.00 | 5.50 | 2.62 |
| typo | -- | -- | -- | 3.00 | 3.25 | 1.00 | 4.67 | -- | -- | -- | 3.00 | -- | -- | 2.50 | 1.75 | 1.47 |

| set | L16 | L17 | L18 | L19 | L20 | L21 | L22 | L23 | L24 | L25 | L26 | L27 | L28 | L29 | L30 | L31 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multihop | 2.33 | 1.87 | 2.33 | 1.67 | 2.70 | 1.34 | 1.64 | 2.14 | 1.00 | 2.42 | 1.00 | 1.20 | 1.15 | 1.06 | 1.05 | 1.08 |
| multilingual | 3.81 | 1.95 | 1.91 | 2.01 | 1.41 | 1.04 | 1.01 | 1.20 | 1.00 | 1.17 | 1.53 | 1.25 | 1.08 | 1.04 | 1.02 | 1.04 |
| poetry | 1.00 | 1.00 | 0.89 | -- | 1.11 | 0.75 | 1.00 | 1.13 | 1.00 | 1.00 | 1.67 | 1.25 | 0.70 | 0.40 | 0.67 | 0.80 |
| order-ops | 6.00 | 2.23 | 4.33 | 3.36 | 1.35 | 1.43 | 1.07 | 1.19 | 2.00 | 10.00 | 0.96 | 1.03 | 1.29 | 1.05 | 1.22 | 1.33 |
| association | 3.00 | 1.27 | 15.00 | 2.50 | 1.00 | 1.13 | 1.25 | 2.14 | 0.71 | 1.00 | 1.83 | 0.60 | 1.00 | 1.06 | 0.67 | 1.15 |
| typo | 1.28 | 0.96 | 1.59 | 1.43 | 1.38 | 1.04 | 1.06 | 1.05 | 1.09 | 1.27 | 1.00 | 1.03 | 1.06 | 0.88 | 1.04 | 1.06 |


**Best-site agreement**, J-head and Logit-head: fraction of (item, intermediate) pairs whose best site is the same under both readouts

| set | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multihop | 0.01 | 0.04 | 0.00 | 0.11 | 0.02 | 0.08 | 0.04 | 0.04 | 0.13 | 0.03 | 0.06 | 0.10 | 0.12 | 0.07 | 0.16 | 0.23 |
| multilingual | 0.02 | 0.04 | 0.04 | 0.04 | 0.04 | 0.05 | 0.12 | 0.06 | 0.08 | 0.07 | 0.04 | 0.05 | 0.12 | 0.15 | 0.37 | 0.33 |
| poetry | 0.02 | 0.03 | 0.01 | 0.07 | 0.06 | 0.05 | 0.11 | 0.04 | 0.08 | 0.04 | 0.09 | 0.06 | 0.10 | 0.10 | 0.09 | 0.19 |
| order-ops | 0.01 | 0.05 | 0.04 | 0.06 | 0.03 | 0.02 | 0.12 | 0.02 | 0.04 | 0.06 | 0.11 | 0.00 | 0.08 | 0.05 | 0.04 | 0.20 |
| association | 0.02 | 0.04 | 0.02 | 0.00 | 0.05 | 0.09 | 0.07 | 0.01 | 0.09 | 0.06 | 0.05 | 0.03 | 0.07 | 0.10 | 0.23 | 0.25 |
| typo | 0.02 | 0.03 | 0.05 | 0.09 | 0.18 | 0.10 | 0.12 | 0.04 | 0.11 | 0.04 | 0.08 | 0.08 | 0.17 | 0.16 | 0.36 | 0.51 |

| set | L16 | L17 | L18 | L19 | L20 | L21 | L22 | L23 | L24 | L25 | L26 | L27 | L28 | L29 | L30 | L31 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multihop | 0.18 | 0.34 | 0.49 | 0.34 | 0.34 | 0.44 | 0.61 | 0.49 | 0.48 | 0.37 | 0.43 | 0.56 | 0.64 | 0.43 | 0.53 | 0.71 |
| multilingual | 0.33 | 0.50 | 0.42 | 0.55 | 0.52 | 0.51 | 0.66 | 0.62 | 0.65 | 0.52 | 0.45 | 0.58 | 0.64 | 0.49 | 0.68 | 0.75 |
| poetry | 0.18 | 0.24 | 0.49 | 0.26 | 0.50 | 0.45 | 0.54 | 0.44 | 0.46 | 0.46 | 0.50 | 0.24 | 0.45 | 0.42 | 0.51 | 0.59 |
| order-ops | 0.15 | 0.27 | 0.31 | 0.47 | 0.45 | 0.37 | 0.58 | 0.53 | 0.14 | 0.30 | 0.39 | 0.45 | 0.55 | 0.68 | 0.39 | 0.55 |
| association | 0.26 | 0.39 | 0.21 | 0.24 | 0.25 | 0.43 | 0.49 | 0.55 | 0.41 | 0.36 | 0.49 | 0.39 | 0.46 | 0.42 | 0.49 | 0.54 |
| typo | 0.40 | 0.53 | 0.45 | 0.68 | 0.69 | 0.73 | 0.88 | 0.84 | 0.72 | 0.61 | 0.70 | 0.66 | 0.85 | 0.66 | 0.82 | 0.85 |

| set | all layers |
|---|---|
| multihop | 0.63 |
| multilingual | 0.83 |
| poetry | 0.51 |
| order-ops | 0.67 |
| association | 0.34 |
| typo | 1.00 |

## 2-3. The coordinate is load-bearing, and how many sites carry it
*One independent variable: which sites are edited. Same trials, same intervention, same arms throughout. At each site count three rows edit the same number of sites and differ only in which: the highest-loading on the source concept (`top_k`), a random draw (`random_k`), the lowest-loading (`bottom_k`); `active` is the per-site-null threshold set, whose count is an outcome, with its two controls drawn at that count.*

**verbal-report** -- 140 trials

The greedy answer is a member of its category on 130 of 140 trials; elsewhere it is ' "'. Where it is not, the direction swapped out is not a concept direction and success means the target overtakes that token.

`layer_wide`, the paper's own intervention -- the Jacobian lens on the residual stream at every block 0-30 -- is the reference the head-level rows are read against: head_jlens a1 0.571; head_jlens a2 0.736; matched_norm a1 0.000; matched_norm a2 0.014. It is block-anchored through the Jacobian lens, not `zin` through `J^z`: a second patch family, marked rather than blended in. Without it a null at any count would say nothing about the count and everything about the setup.

`head_jlens` arm, swap rate by site count

| sites | top a1 | random a1 | bottom a1 | top a2 | random a2 | bottom a2 |
|---|---|---|---|---|---|---|
| 1 | 0.007 | 0.000 | 0.000 | 0.021 | 0.000 | 0.000 |
| 8 | 0.129 | 0.000 | 0.000 | 0.336 | 0.000 | 0.000 |
| 16 | 0.229 | 0.000 | 0.000 | 0.457 | 0.000 | 0.000 |
| 32 | 0.321 | 0.000 | 0.007 | 0.664 | 0.000 | 0.007 |
| 64 | 0.400 | 0.000 | 0.000 | 0.764 | 0.021 | 0.000 |
| active (median 49.5) | 0.443 | 0.000 | 0.000 | 0.757 | 0.014 | 0.000 |

`matched_norm` arm, swap rate by site count

| sites | top a1 | random a1 | bottom a1 | top a2 | random a2 | bottom a2 |
|---|---|---|---|---|---|---|
| 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 32 | 0.000 | 0.000 | 0.000 | 0.007 | 0.000 | 0.000 |
| 64 | 0.000 | 0.000 | 0.000 | 0.007 | 0.007 | 0.000 |
| active (median 49.5) | 0.000 | 0.000 | 0.000 | 0.007 | 0.007 | 0.000 |

Selection descriptives. A concept is active at 52.7 of 1024 sites per trial (median 49.5, range 33-72, 0 trials with none), against a per-site 99th-percentile null. Per count, top / random / bottom: mean loading of the selected sites, their mean relative write norm (from read.py), and the total edit norm the `head_jlens` edit realised at alpha=1. The controls are matched in count, not in size -- high loading implies a large `|c_s|` implies a large edit -- so the last column is the size difference left visible rather than equalised away.

| sites | mean loading | relative write norm | total edit norm |
|---|---|---|---|
| 1 | 0.633 / 0.011 / -0.566 | 0.039 / 0.047 / 0.035 | 1.3 / 0.3 / 0.6 |
| 8 | 0.506 / 0.010 / -0.436 | 0.040 / 0.048 / 0.039 | 3.5 / 1.3 / 2.3 |
| 16 | 0.439 / 0.009 / -0.378 | 0.043 / 0.047 / 0.039 | 3.9 / 1.8 / 2.9 |
| 32 | 0.368 / 0.010 / -0.310 | 0.044 / 0.046 / 0.043 | 4.8 / 2.6 / 3.7 |
| 64 | 0.295 / 0.009 / -0.248 | 0.045 / 0.047 / 0.047 | 5.6 / 3.8 / 4.6 |
| active (median 49.5) | 0.289 / 0.010 / -0.267 | 0.040 / 0.047 / 0.046 | 6.7 / 3.4 / 4.4 |

The concept-selected sets span (distinct layers / distinct (layer, GQA-group) pairs): `top_1` 1.0/1.0, `top_8` 6.1/6.9, `top_16` 10.6/13.1, `top_32` 17.0/25.6, `top_64` 25.2/51.6, `active` 21.9/41.9 -- a set that kept picking group-mates would not be that many independent sites, since one `W_V` is shared within a GQA group.

**probe-swap** -- 45 trials

`layer_wide`, the paper's own intervention -- the Jacobian lens on the residual stream at every block 0-30 -- is the reference the head-level rows are read against: head_jlens a1 0.444; head_jlens a2 0.244; matched_norm a1 0.044; matched_norm a2 0.044. It is block-anchored through the Jacobian lens, not `zin` through `J^z`: a second patch family, marked rather than blended in. Without it a null at any count would say nothing about the count and everything about the setup.

`head_jlens` arm, swap rate by site count

| sites | top a1 | random a1 | bottom a1 | top a2 | random a2 | bottom a2 |
|---|---|---|---|---|---|---|
| 1 | 0.000 | 0.000 | 0.000 | 0.178 | 0.000 | 0.000 |
| 8 | 0.333 | 0.000 | 0.000 | 0.511 | 0.000 | 0.000 |
| 16 | 0.467 | 0.000 | 0.000 | 0.511 | 0.000 | 0.000 |
| 32 | 0.489 | 0.000 | 0.000 | 0.556 | 0.000 | 0.022 |
| 64 | 0.511 | 0.000 | 0.000 | 0.556 | 0.044 | 0.022 |
| active (median 79) | 0.511 | 0.000 | 0.000 | 0.511 | 0.067 | 0.022 |

`matched_norm` arm, swap rate by site count

| sites | top a1 | random a1 | bottom a1 | top a2 | random a2 | bottom a2 |
|---|---|---|---|---|---|---|
| 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 64 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| active (median 79) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Selection descriptives. A concept is active at 73.9 of 1024 sites per trial (median 79, range 32-119, 0 trials with none), against a per-site 99th-percentile null. Per count, top / random / bottom: mean loading of the selected sites, their mean relative write norm (from read.py), and the total edit norm the `head_jlens` edit realised at alpha=1. The controls are matched in count, not in size -- high loading implies a large `|c_s|` implies a large edit -- so the last column is the size difference left visible rather than equalised away.

| sites | mean loading | relative write norm | total edit norm |
|---|---|---|---|
| 1 | 0.686 / 0.042 / -0.586 | 0.033 / 0.053 / 0.042 | 2.8 / 0.4 / 0.9 |
| 8 | 0.553 / 0.024 / -0.452 | 0.039 / 0.046 / 0.047 | 5.0 / 1.5 / 3.1 |
| 16 | 0.490 / 0.020 / -0.393 | 0.042 / 0.047 / 0.050 | 5.7 / 2.3 / 4.1 |
| 32 | 0.415 / 0.017 / -0.332 | 0.044 / 0.046 / 0.052 | 6.6 / 3.2 / 5.4 |
| 64 | 0.335 / 0.014 / -0.270 | 0.048 / 0.047 / 0.052 | 7.5 / 4.9 / 6.7 |
| active (median 79) | 0.300 / 0.014 / -0.258 | 0.047 / 0.047 / 0.052 | 7.8 / 5.2 / 7.0 |

The concept-selected sets span (distinct layers / distinct (layer, GQA-group) pairs): `top_1` 1.0/1.0, `top_8` 6.8/7.1, `top_16` 10.9/12.9, `top_32` 17.4/24.8, `top_64` 25.3/49.1, `active` 25.0/56.2 -- a set that kept picking group-mates would not be that many independent sites, since one `W_V` is shared within a GQA group.

alpha=1 exchanges the two lens coordinates; alpha=2 extrapolates past the exchange (the v_s coordinate becomes 2c_1 - c_0). They are different operations, not two strengths of one. Read both, never their maximum.

## What this does not establish
- **Legibility is not magnitude.** RMSNorm is scale-invariant, so a head contributing almost nothing still yields a confident-looking top-1. The heads above write a median 3.9% of the residual stream they join.
- **probe-swap deviates.** Its items substitute lens directions for the shipped linear-probe directions, and they are not the paper's headline 50-prompt set, so those rates are not comparable to the paper's.
