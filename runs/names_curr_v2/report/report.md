# Looped-head depth sweep

Base model: `HuggingFaceTB/SmolLM2-360M-Instruct`

## Accuracy: test (n=600)

| head | R | accuracy | 95% CI | vs R=0 | p | valid JSON | 1 hops | 2 hops | 3 hops |
|---|---|---|---|---|---|---|---|---|---|
| frozen baseline | 0 | 0.140 | 0.115-0.170 |  |  | 0.675 | 0.382 | 0.049 | 0.005 |
| 1x-plain | 1* | 0.485 | 0.445-0.525 | +0.345 | 3.7e-56 | 1.000 | 0.969 | 0.335 | 0.184 |
| 1x-plain | 2† | 0.478 | 0.439-0.518 | +0.338 | 5e-54 | 1.000 | 0.979 | 0.310 | 0.180 |
| 1x-plain | 3† | 0.455 | 0.416-0.495 | +0.315 | 4.6e-46 | 0.998 | 0.921 | 0.305 | 0.170 |
| 3x-chain | 1 | 0.308 | 0.273-0.346 | +0.168 | 2.5e-21 | 1.000 | 0.969 | 0.000 | 0.000 |
| 3x-chain | 2 | 0.447 | 0.407-0.487 | +0.307 | 1.5e-48 | 1.000 | 0.974 | 0.404 | 0.000 |
| 3x-chain | 3* | 0.533 | 0.493-0.573 | +0.393 | 1.3e-64 | 1.000 | 0.974 | 0.424 | 0.233 |

`*` trained depth, `†` deeper than the head was trained. p: exact McNemar vs R=0.

## Acceptance

| head | criterion | result | detail |
|---|---|---|---|
| 1x-plain | beats frozen baseline | PASS | R=1: 0.485 vs R=0: 0.140 (gain +0.345, p=3.7e-56) |
| 1x-plain | monotonic up to trained depth | PASS | R0:0.140 -> R1:0.485 |
| 1x-plain | stable beyond trained depth | FAIL (info) | worst at R=3: 0.455 vs R=1: 0.485 |
| 1x-plain | **verdict** | **PASS** | required criteria only |
| 3x-chain | beats frozen baseline | PASS | R=3: 0.533 vs R=0: 0.140 (gain +0.393, p=1.3e-64) |
| 3x-chain | monotonic up to trained depth | PASS (info) | R0:0.140 -> R1:0.308 -> R2:0.447 -> R3:0.533 | head was trained at a single depth; shallower depths are off-distribution |
| 3x-chain | looping beats single-pass control | PASS | R=3: 0.533 vs '1x-plain' at R=1: 0.485 (p=0.007) |
| 3x-chain | deeper beats own first pass | PASS (info) | R=3: 0.533 vs R=1: 0.308 (p=1.6e-39) |
| 3x-chain | **verdict** | **PASS** | required criteria only |
