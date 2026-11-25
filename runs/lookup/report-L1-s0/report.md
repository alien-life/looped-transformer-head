# Looped-head depth sweep

Base model: `HuggingFaceTB/SmolLM2-360M-Instruct`

## Accuracy: test (n=600)

| head | R | accuracy | 95% CI | vs R=0 | p | valid JSON | 1 hops | 2 hops | 3 hops |
|---|---|---|---|---|---|---|---|---|---|
| frozen baseline | 0 | 0.322 | 0.286-0.360 |  |  | 1.000 | 0.641 | 0.282 | 0.000 |
| 1x | 1* | 0.427 | 0.388-0.467 | +0.105 | 7e-08 | 1.000 | 0.699 | 0.306 | 0.253 |
| 1x | 2† | 0.372 | 0.334-0.411 | +0.050 | 0.021 | 1.000 | 0.598 | 0.273 | 0.225 |
| 2x | 1 | 0.437 | 0.398-0.477 | +0.115 | 1.9e-14 | 1.000 | 0.713 | 0.282 | 0.297 |
| 2x | 2* | 0.452 | 0.412-0.492 | +0.130 | 1.7e-10 | 1.000 | 0.703 | 0.316 | 0.319 |

`*` trained depth, `†` deeper than the head was trained. p: exact McNemar vs R=0.

## Acceptance

| head | criterion | result | detail |
|---|---|---|---|
| 1x | beats frozen baseline | PASS | R=1: 0.427 vs R=0: 0.322 (gain +0.105, p=7e-08) |
| 1x | monotonic up to trained depth | PASS | R0:0.322 -> R1:0.427 |
| 1x | stable beyond trained depth | FAIL (info) | worst at R=2: 0.372 vs R=1: 0.427 |
| 1x | **verdict** | **PASS** | required criteria only |
| 2x | beats frozen baseline | PASS | R=2: 0.452 vs R=0: 0.322 (gain +0.130, p=1.7e-10) |
| 2x | monotonic up to trained depth | PASS (info) | R0:0.322 -> R1:0.437 -> R2:0.452 | head was trained at a single depth; shallower depths are off-distribution |
| 2x | looping beats single-pass control | FAIL | R=2: 0.452 vs '1x' at R=1: 0.427 (p=0.12) |
| 2x | deeper beats own first pass | FAIL (info) | R=2: 0.452 vs R=1: 0.437 (p=0.51) |
| 2x | **verdict** | **FAIL** | required criteria only |
