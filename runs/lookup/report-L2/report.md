# Looped-head depth sweep

Base model: `HuggingFaceTB/SmolLM2-360M-Instruct`

## Accuracy: test (n=600)

| head | R | accuracy | 95% CI | vs R=0 | p | valid JSON | 1 hops | 2 hops | 3 hops |
|---|---|---|---|---|---|---|---|---|---|
| frozen baseline | 0 | 0.322 | 0.286-0.360 |  |  | 1.000 | 0.641 | 0.282 | 0.000 |
| 1x-L2-s0 | 1* | 0.448 | 0.409-0.488 | +0.127 | 1.2e-09 | 1.000 | 0.766 | 0.301 | 0.253 |
| 1x-L2-s0 | 2† | 0.435 | 0.396-0.475 | +0.113 | 4.4e-07 | 1.000 | 0.713 | 0.301 | 0.269 |
| 1x-L2-s0 | 3† | 0.192 | 0.162-0.225 | -0.130 | 3.4e-07 | 0.982 | 0.081 | 0.278 | 0.220 |
| 2x-L2-s0 | 1 | 0.460 | 0.421-0.500 | +0.138 | 1.7e-13 | 1.000 | 0.751 | 0.297 | 0.313 |
| 2x-L2-s0 | 2* | 0.478 | 0.439-0.518 | +0.157 | 6.5e-13 | 1.000 | 0.794 | 0.311 | 0.308 |
| 2x-L2-s0 | 3† | 0.447 | 0.407-0.487 | +0.125 | 3.5e-08 | 1.000 | 0.761 | 0.306 | 0.247 |
| 3x-L2-s0 | 1 | 0.448 | 0.409-0.488 | +0.127 | 5.5e-11 | 1.000 | 0.713 | 0.335 | 0.275 |
| 3x-L2-s0 | 2 | 0.455 | 0.416-0.495 | +0.133 | 2.7e-11 | 1.000 | 0.737 | 0.330 | 0.275 |
| 3x-L2-s0 | 3* | 0.445 | 0.406-0.485 | +0.123 | 8.7e-10 | 1.000 | 0.732 | 0.316 | 0.264 |

`*` trained depth, `†` deeper than the head was trained. p: exact McNemar vs R=0.

## Acceptance

| head | criterion | result | detail |
|---|---|---|---|
| 1x-L2-s0 | beats frozen baseline | PASS | R=1: 0.448 vs R=0: 0.322 (gain +0.127, p=1.2e-09) |
| 1x-L2-s0 | monotonic up to trained depth | PASS | R0:0.322 -> R1:0.448 |
| 1x-L2-s0 | stable beyond trained depth | FAIL (info) | worst at R=3: 0.192 vs R=1: 0.448 |
| 1x-L2-s0 | **verdict** | **PASS** | required criteria only |
| 2x-L2-s0 | beats frozen baseline | PASS | R=2: 0.478 vs R=0: 0.322 (gain +0.157, p=6.5e-13) |
| 2x-L2-s0 | monotonic up to trained depth | PASS (info) | R0:0.322 -> R1:0.460 -> R2:0.478 | head was trained at a single depth; shallower depths are off-distribution |
| 2x-L2-s0 | looping beats single-pass control | FAIL | R=2: 0.478 vs '1x-L2-s0' at R=1: 0.448 (p=0.14) |
| 2x-L2-s0 | deeper beats own first pass | FAIL (info) | R=2: 0.478 vs R=1: 0.460 (p=0.29) |
| 2x-L2-s0 | stable beyond trained depth | FAIL (info) | worst at R=3: 0.447 vs R=2: 0.478 |
| 2x-L2-s0 | **verdict** | **FAIL** | required criteria only |
| 3x-L2-s0 | beats frozen baseline | PASS | R=3: 0.445 vs R=0: 0.322 (gain +0.123, p=8.7e-10) |
| 3x-L2-s0 | monotonic up to trained depth | PASS (info) | R0:0.322 -> R1:0.448 -> R2:0.455 -> R3:0.445 | head was trained at a single depth; shallower depths are off-distribution |
| 3x-L2-s0 | looping beats single-pass control | FAIL | R=3: 0.445 vs '1x-L2-s0' at R=1: 0.448 (p=0.93) |
| 3x-L2-s0 | deeper beats own first pass | FAIL (info) | R=3: 0.445 vs R=1: 0.448 (p=0.89) |
| 3x-L2-s0 | **verdict** | **FAIL** | required criteria only |
