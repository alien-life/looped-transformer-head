# Per-loop decoding probe

## frozen backbone

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 0 | answer | 0.126 |  |  |  |  |
| 2 | 203 | 0 | answer | 0.010 | 0.015 | 0.044 |  | 0.941 |
| 3 | 206 | 0 | answer | 0.000 | 0.019 | 0.000 | 0.005 | 0.976 |

## 1x-plain (R=1, final answer only)

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 1 | answer | 0.576 |  |  |  |  |
| 2 | 203 | 1 | answer | 0.320 | 0.000 | 0.320 |  | 0.680 |
| 3 | 206 | 1 | answer | 0.209 | 0.000 | 0.000 | 0.209 | 0.791 |

## 3x-chain (R=3, chain-supervised)

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 1 | answer | 0.555 |  |  |  |  |
| 1 | 191 | 2 | answer | 0.550 |  |  |  |  |
| 1 | 191 | 3 | answer | 0.576 |  |  |  |  |
| 2 | 203 | 1 | hop 1 | 0.305 | 0.305 | 0.000 |  | 0.695 |
| 2 | 203 | 2 | answer | 0.286 | 0.000 | 0.286 |  | 0.714 |
| 2 | 203 | 3 | answer | 0.315 | 0.000 | 0.315 |  | 0.685 |
| 3 | 206 | 1 | hop 1 | 0.447 | 0.447 | 0.000 | 0.000 | 0.553 |
| 3 | 206 | 2 | hop 2 | 0.301 | 0.000 | 0.301 | 0.000 | 0.699 |
| 3 | 206 | 3 | answer | 0.199 | 0.000 | 0.005 | 0.199 | 0.796 |

`holds hop j`: share of examples whose top prediction at the entity token is hop j's entity (`other`: anything else). Loop 0 is the frozen backbone.
