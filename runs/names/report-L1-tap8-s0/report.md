# Looped-head depth sweep

Base model: `HuggingFaceTB/SmolLM2-360M-Instruct`

## Accuracy: test (n=600)

| head | R | accuracy | 95% CI | vs R=0 | p | valid JSON | 1 hops | 2 hops | 3 hops |
|---|---|---|---|---|---|---|---|---|---|
| frozen baseline | 0 | 0.140 | 0.115-0.170 |  |  | 0.675 | 0.382 | 0.049 | 0.005 |
| 1x-plain-L1-tap8-s0 | 1* | 0.350 | 0.313-0.389 | +0.210 | 7.8e-25 | 1.000 | 0.592 | 0.291 | 0.184 |
| 1x-plain-L1-tap8-s0 | 2† | 0.283 | 0.249-0.321 | +0.143 | 1.3e-09 | 1.000 | 0.403 | 0.291 | 0.165 |
| 1x-plain-L1-tap8-s0 | 3† | 0.263 | 0.230-0.300 | +0.123 | 4.1e-07 | 1.000 | 0.335 | 0.305 | 0.155 |
| 3x-plain-L1-tap8-s0 | 1 | 0.255 | 0.222-0.291 | +0.115 | 9.4e-16 | 1.000 | 0.492 | 0.251 | 0.039 |
| 3x-plain-L1-tap8-s0 | 2 | 0.330 | 0.294-0.369 | +0.190 | 3.9e-25 | 1.000 | 0.524 | 0.296 | 0.184 |
| 3x-plain-L1-tap8-s0 | 3* | 0.363 | 0.326-0.403 | +0.223 | 8.2e-27 | 1.000 | 0.592 | 0.300 | 0.214 |
| 2x-chain-L1-tap8-s0 | 1 | 0.190 | 0.161-0.223 | +0.050 | 0.0016 | 1.000 | 0.597 | 0.000 | 0.000 |
| 2x-chain-L1-tap8-s0 | 2* | 0.347 | 0.310-0.386 | +0.207 | 2.4e-24 | 1.000 | 0.586 | 0.281 | 0.189 |
| 2x-chain-L1-tap8-s0 | 3† | 0.338 | 0.302-0.377 | +0.198 | 8.9e-23 | 1.000 | 0.592 | 0.291 | 0.150 |
| 3x-chain-L1-tap8-s0 | 1 | 0.188 | 0.159-0.222 | +0.048 | 0.0022 | 1.000 | 0.592 | 0.000 | 0.000 |
| 3x-chain-L1-tap8-s0 | 2 | 0.280 | 0.246-0.317 | +0.140 | 7e-15 | 1.000 | 0.592 | 0.271 | 0.000 |
| 3x-chain-L1-tap8-s0 | 3* | 0.343 | 0.306-0.382 | +0.203 | 1.3e-24 | 1.000 | 0.586 | 0.271 | 0.189 |

`*` trained depth, `†` deeper than the head was trained. p: exact McNemar vs R=0.

## Acceptance

| head | criterion | result | detail |
|---|---|---|---|
| 1x-plain-L1-tap8-s0 | beats frozen baseline | PASS | R=1: 0.350 vs R=0: 0.140 (gain +0.210, p=7.8e-25) |
| 1x-plain-L1-tap8-s0 | monotonic up to trained depth | PASS | R0:0.140 -> R1:0.350 |
| 1x-plain-L1-tap8-s0 | stable beyond trained depth | FAIL (info) | worst at R=3: 0.263 vs R=1: 0.350 |
| 1x-plain-L1-tap8-s0 | **verdict** | **PASS** | required criteria only |
| 3x-plain-L1-tap8-s0 | beats frozen baseline | PASS | R=3: 0.363 vs R=0: 0.140 (gain +0.223, p=8.2e-27) |
| 3x-plain-L1-tap8-s0 | monotonic up to trained depth | PASS (info) | R0:0.140 -> R1:0.255 -> R2:0.330 -> R3:0.363 | head was trained at a single depth; shallower depths are off-distribution |
| 3x-plain-L1-tap8-s0 | looping beats single-pass control | FAIL | R=3: 0.363 vs '1x-plain-L1-tap8-s0' at R=1: 0.350 (p=0.17) |
| 3x-plain-L1-tap8-s0 | deeper beats own first pass | PASS (info) | R=3: 0.363 vs R=1: 0.255 (p=8.4e-10) |
| 3x-plain-L1-tap8-s0 | **verdict** | **FAIL** | required criteria only |
| 2x-chain-L1-tap8-s0 | beats frozen baseline | PASS | R=2: 0.347 vs R=0: 0.140 (gain +0.207, p=2.4e-24) |
| 2x-chain-L1-tap8-s0 | monotonic up to trained depth | PASS (info) | R0:0.140 -> R1:0.190 -> R2:0.347 | head was trained at a single depth; shallower depths are off-distribution |
| 2x-chain-L1-tap8-s0 | looping beats single-pass control | FAIL | R=2: 0.347 vs '1x-plain-L1-tap8-s0' at R=1: 0.350 (p=0.87) |
| 2x-chain-L1-tap8-s0 | deeper beats own first pass | PASS (info) | R=2: 0.347 vs R=1: 0.190 (p=1.7e-24) |
| 2x-chain-L1-tap8-s0 | stable beyond trained depth | PASS (info) | worst at R=3: 0.338 vs R=2: 0.347 |
| 2x-chain-L1-tap8-s0 | **verdict** | **FAIL** | required criteria only |
| 3x-chain-L1-tap8-s0 | beats frozen baseline | PASS | R=3: 0.343 vs R=0: 0.140 (gain +0.203, p=1.3e-24) |
| 3x-chain-L1-tap8-s0 | monotonic up to trained depth | PASS (info) | R0:0.140 -> R1:0.188 -> R2:0.280 -> R3:0.343 | head was trained at a single depth; shallower depths are off-distribution |
| 3x-chain-L1-tap8-s0 | looping beats single-pass control | FAIL | R=3: 0.343 vs '1x-plain-L1-tap8-s0' at R=1: 0.350 (p=0.57) |
| 3x-chain-L1-tap8-s0 | deeper beats own first pass | PASS (info) | R=3: 0.343 vs R=1: 0.188 (p=3.4e-24) |
| 3x-chain-L1-tap8-s0 | **verdict** | **FAIL** | required criteria only |
