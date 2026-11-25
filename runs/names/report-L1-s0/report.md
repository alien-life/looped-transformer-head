# Looped-head depth sweep

Base model: `HuggingFaceTB/SmolLM2-360M-Instruct`

## Accuracy: test (n=600)

| head | R | accuracy | 95% CI | vs R=0 | p | valid JSON | 1 hops | 2 hops | 3 hops |
|---|---|---|---|---|---|---|---|---|---|
| frozen baseline | 0 | 0.140 | 0.115-0.170 |  |  | 0.675 | 0.382 | 0.049 | 0.005 |
| 1x-plain-L1-s0 | 1* | 0.377 | 0.339-0.416 | +0.237 | 4.1e-25 | 1.000 | 0.581 | 0.340 | 0.223 |
| 1x-plain-L1-s0 | 2† | 0.237 | 0.204-0.272 | +0.097 | 1.8e-05 | 1.000 | 0.346 | 0.281 | 0.092 |
| 1x-plain-L1-s0 | 3† | 0.123 | 0.099-0.152 | -0.017 | 0.44 | 0.995 | 0.230 | 0.089 | 0.058 |
| 3x-plain-L1-s0 | 1 | 0.278 | 0.244-0.316 | +0.138 | 1.8e-18 | 1.000 | 0.524 | 0.300 | 0.029 |
| 3x-plain-L1-s0 | 2 | 0.343 | 0.306-0.382 | +0.203 | 9.4e-27 | 1.000 | 0.565 | 0.300 | 0.180 |
| 3x-plain-L1-s0 | 3* | 0.353 | 0.316-0.392 | +0.213 | 5.8e-25 | 1.000 | 0.602 | 0.286 | 0.189 |
| 2x-chain-L1-s0 | 1 | 0.177 | 0.148-0.209 | +0.037 | 0.03 | 1.000 | 0.555 | 0.000 | 0.000 |
| 2x-chain-L1-s0 | 2* | 0.347 | 0.310-0.386 | +0.207 | 1.5e-21 | 1.000 | 0.539 | 0.300 | 0.214 |
| 2x-chain-L1-s0 | 3† | 0.333 | 0.297-0.372 | +0.193 | 9.4e-20 | 1.000 | 0.545 | 0.305 | 0.165 |
| 3x-chain-L1-s0 | 1 | 0.182 | 0.153-0.215 | +0.042 | 0.013 | 1.000 | 0.571 | 0.000 | 0.000 |
| 3x-chain-L1-s0 | 2 | 0.297 | 0.262-0.334 | +0.157 | 1.2e-15 | 1.000 | 0.592 | 0.320 | 0.000 |
| 3x-chain-L1-s0 | 3* | 0.372 | 0.334-0.411 | +0.232 | 3e-26 | 1.000 | 0.592 | 0.310 | 0.228 |

`*` trained depth, `†` deeper than the head was trained. p: exact McNemar vs R=0.

## Acceptance

| head | criterion | result | detail |
|---|---|---|---|
| 1x-plain-L1-s0 | beats frozen baseline | PASS | R=1: 0.377 vs R=0: 0.140 (gain +0.237, p=4.1e-25) |
| 1x-plain-L1-s0 | monotonic up to trained depth | PASS | R0:0.140 -> R1:0.377 |
| 1x-plain-L1-s0 | stable beyond trained depth | FAIL (info) | worst at R=3: 0.123 vs R=1: 0.377 |
| 1x-plain-L1-s0 | **verdict** | **PASS** | required criteria only |
| 3x-plain-L1-s0 | beats frozen baseline | PASS | R=3: 0.353 vs R=0: 0.140 (gain +0.213, p=5.8e-25) |
| 3x-plain-L1-s0 | monotonic up to trained depth | PASS (info) | R0:0.140 -> R1:0.278 -> R2:0.343 -> R3:0.353 | head was trained at a single depth; shallower depths are off-distribution |
| 3x-plain-L1-s0 | looping beats single-pass control | FAIL | R=3: 0.353 vs '1x-plain-L1-s0' at R=1: 0.377 (p=0.18) |
| 3x-plain-L1-s0 | deeper beats own first pass | PASS (info) | R=3: 0.353 vs R=1: 0.278 (p=7e-05) |
| 3x-plain-L1-s0 | **verdict** | **FAIL** | required criteria only |
| 2x-chain-L1-s0 | beats frozen baseline | PASS | R=2: 0.347 vs R=0: 0.140 (gain +0.207, p=1.5e-21) |
| 2x-chain-L1-s0 | monotonic up to trained depth | PASS (info) | R0:0.140 -> R1:0.177 -> R2:0.347 | head was trained at a single depth; shallower depths are off-distribution |
| 2x-chain-L1-s0 | looping beats single-pass control | FAIL | R=2: 0.347 vs '1x-plain-L1-s0' at R=1: 0.377 (p=0.06) |
| 2x-chain-L1-s0 | deeper beats own first pass | PASS (info) | R=2: 0.347 vs R=1: 0.177 (p=5.4e-26) |
| 2x-chain-L1-s0 | stable beyond trained depth | FAIL (info) | worst at R=3: 0.333 vs R=2: 0.347 |
| 2x-chain-L1-s0 | **verdict** | **FAIL** | required criteria only |
| 3x-chain-L1-s0 | beats frozen baseline | PASS | R=3: 0.372 vs R=0: 0.140 (gain +0.232, p=3e-26) |
| 3x-chain-L1-s0 | monotonic up to trained depth | PASS (info) | R0:0.140 -> R1:0.182 -> R2:0.297 -> R3:0.372 | head was trained at a single depth; shallower depths are off-distribution |
| 3x-chain-L1-s0 | looping beats single-pass control | FAIL | R=3: 0.372 vs '1x-plain-L1-s0' at R=1: 0.377 (p=0.81) |
| 3x-chain-L1-s0 | deeper beats own first pass | PASS (info) | R=3: 0.372 vs R=1: 0.182 (p=1.2e-28) |
| 3x-chain-L1-s0 | **verdict** | **FAIL** | required criteria only |
