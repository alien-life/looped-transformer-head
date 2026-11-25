# Looped-head depth sweep

Base model: `HuggingFaceTB/SmolLM2-360M-Instruct`

## Accuracy: test (n=600)

| head | R | accuracy | 95% CI | vs R=0 | p | valid JSON | 1 hops | 2 hops | 3 hops |
|---|---|---|---|---|---|---|---|---|---|
| frozen baseline | 0 | 0.140 | 0.115-0.170 |  |  | 0.675 | 0.382 | 0.049 | 0.005 |
| 1x-plain-L2-s0 | 1* | 0.355 | 0.318-0.394 | +0.215 | 7.2e-24 | 1.000 | 0.576 | 0.305 | 0.199 |
| 1x-plain-L2-s0 | 2† | 0.298 | 0.263-0.336 | +0.158 | 7.1e-11 | 1.000 | 0.419 | 0.330 | 0.155 |
| 1x-plain-L2-s0 | 3† | 0.267 | 0.233-0.303 | +0.127 | 2.5e-07 | 1.000 | 0.351 | 0.320 | 0.136 |
| 3x-plain-L2-s0 | 1 | 0.215 | 0.184-0.250 | +0.075 | 1.4e-09 | 0.997 | 0.461 | 0.167 | 0.034 |
| 3x-plain-L2-s0 | 2 | 0.333 | 0.297-0.372 | +0.193 | 3.6e-26 | 1.000 | 0.518 | 0.305 | 0.189 |
| 3x-plain-L2-s0 | 3* | 0.353 | 0.316-0.392 | +0.213 | 2.5e-25 | 1.000 | 0.592 | 0.305 | 0.180 |
| 2x-chain-L2-s0 | 1 | 0.190 | 0.161-0.223 | +0.050 | 0.0021 | 1.000 | 0.597 | 0.000 | 0.000 |
| 2x-chain-L2-s0 | 2* | 0.355 | 0.318-0.394 | +0.215 | 1.4e-25 | 1.000 | 0.597 | 0.291 | 0.194 |
| 2x-chain-L2-s0 | 3† | 0.347 | 0.310-0.386 | +0.207 | 5.5e-24 | 1.000 | 0.550 | 0.305 | 0.199 |
| 3x-chain-L2-s0 | 1 | 0.193 | 0.164-0.227 | +0.053 | 0.00073 | 1.000 | 0.607 | 0.000 | 0.000 |
| 3x-chain-L2-s0 | 2 | 0.290 | 0.255-0.328 | +0.150 | 3.2e-15 | 1.000 | 0.581 | 0.310 | 0.000 |
| 3x-chain-L2-s0 | 3* | 0.360 | 0.323-0.399 | +0.220 | 1.4e-25 | 1.000 | 0.576 | 0.320 | 0.199 |

`*` trained depth, `†` deeper than the head was trained. p: exact McNemar vs R=0.

## Acceptance

| head | criterion | result | detail |
|---|---|---|---|
| 1x-plain-L2-s0 | beats frozen baseline | PASS | R=1: 0.355 vs R=0: 0.140 (gain +0.215, p=7.2e-24) |
| 1x-plain-L2-s0 | monotonic up to trained depth | PASS | R0:0.140 -> R1:0.355 |
| 1x-plain-L2-s0 | stable beyond trained depth | FAIL (info) | worst at R=3: 0.267 vs R=1: 0.355 |
| 1x-plain-L2-s0 | **verdict** | **PASS** | required criteria only |
| 3x-plain-L2-s0 | beats frozen baseline | PASS | R=3: 0.353 vs R=0: 0.140 (gain +0.213, p=2.5e-25) |
| 3x-plain-L2-s0 | monotonic up to trained depth | PASS (info) | R0:0.140 -> R1:0.215 -> R2:0.333 -> R3:0.353 | head was trained at a single depth; shallower depths are off-distribution |
| 3x-plain-L2-s0 | looping beats single-pass control | FAIL | R=3: 0.353 vs '1x-plain-L2-s0' at R=1: 0.355 (p=1) |
| 3x-plain-L2-s0 | deeper beats own first pass | PASS (info) | R=3: 0.353 vs R=1: 0.215 (p=1e-13) |
| 3x-plain-L2-s0 | **verdict** | **FAIL** | required criteria only |
| 2x-chain-L2-s0 | beats frozen baseline | PASS | R=2: 0.355 vs R=0: 0.140 (gain +0.215, p=1.4e-25) |
| 2x-chain-L2-s0 | monotonic up to trained depth | PASS (info) | R0:0.140 -> R1:0.190 -> R2:0.355 | head was trained at a single depth; shallower depths are off-distribution |
| 2x-chain-L2-s0 | looping beats single-pass control | FAIL | R=2: 0.355 vs '1x-plain-L2-s0' at R=1: 0.355 (p=1) |
| 2x-chain-L2-s0 | deeper beats own first pass | PASS (info) | R=2: 0.355 vs R=1: 0.190 (p=3.8e-25) |
| 2x-chain-L2-s0 | stable beyond trained depth | PASS (info) | worst at R=3: 0.347 vs R=2: 0.355 |
| 2x-chain-L2-s0 | **verdict** | **FAIL** | required criteria only |
| 3x-chain-L2-s0 | beats frozen baseline | PASS | R=3: 0.360 vs R=0: 0.140 (gain +0.220, p=1.4e-25) |
| 3x-chain-L2-s0 | monotonic up to trained depth | PASS (info) | R0:0.140 -> R1:0.193 -> R2:0.290 -> R3:0.360 | head was trained at a single depth; shallower depths are off-distribution |
| 3x-chain-L2-s0 | looping beats single-pass control | FAIL | R=3: 0.360 vs '1x-plain-L2-s0' at R=1: 0.355 (p=0.78) |
| 3x-chain-L2-s0 | deeper beats own first pass | PASS (info) | R=3: 0.360 vs R=1: 0.193 (p=1.9e-22) |
| 3x-chain-L2-s0 | **verdict** | **FAIL** | required criteria only |
