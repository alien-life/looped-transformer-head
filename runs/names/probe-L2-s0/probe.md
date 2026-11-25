# Per-loop decoding probe

## frozen backbone

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 0 | answer | 0.126 |  |  |  |  |
| 2 | 203 | 0 | answer | 0.010 | 0.015 | 0.044 |  | 0.941 |
| 3 | 206 | 0 | answer | 0.000 | 0.019 | 0.000 | 0.005 | 0.976 |

## 1x-plain-L2-s0 (R=1, final answer only)

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 1 | answer | 0.571 |  |  |  |  |
| 2 | 203 | 1 | answer | 0.305 | 0.000 | 0.305 |  | 0.695 |
| 3 | 206 | 1 | answer | 0.199 | 0.000 | 0.000 | 0.199 | 0.801 |

## 3x-plain-L2-s0 (R=3, final answer only)

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 1 | answer | 0.461 |  |  |  |  |
| 1 | 191 | 2 | answer | 0.513 |  |  |  |  |
| 1 | 191 | 3 | answer | 0.592 |  |  |  |  |
| 2 | 203 | 1 | hop 1 | 0.015 | 0.015 | 0.167 |  | 0.818 |
| 2 | 203 | 2 | answer | 0.300 | 0.000 | 0.300 |  | 0.700 |
| 2 | 203 | 3 | answer | 0.305 | 0.000 | 0.305 |  | 0.695 |
| 3 | 206 | 1 | hop 1 | 0.044 | 0.044 | 0.015 | 0.034 | 0.908 |
| 3 | 206 | 2 | hop 2 | 0.005 | 0.000 | 0.005 | 0.189 | 0.806 |
| 3 | 206 | 3 | answer | 0.175 | 0.000 | 0.000 | 0.175 | 0.825 |

## 2x-chain-L2-s0 (R=2, chain-supervised)

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 1 | answer | 0.597 |  |  |  |  |
| 1 | 191 | 2 | answer | 0.592 |  |  |  |  |
| 2 | 203 | 1 | hop 1 | 0.330 | 0.330 | 0.000 |  | 0.670 |
| 2 | 203 | 2 | answer | 0.291 | 0.000 | 0.291 |  | 0.709 |
| 3 | 206 | 1 | hop 1 | 0.466 | 0.466 | 0.000 | 0.000 | 0.534 |
| 3 | 206 | 2 | answer | 0.194 | 0.000 | 0.000 | 0.194 | 0.806 |

## 3x-chain-L2-s0 (R=3, chain-supervised)

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 1 | answer | 0.602 |  |  |  |  |
| 1 | 191 | 2 | answer | 0.565 |  |  |  |  |
| 1 | 191 | 3 | answer | 0.576 |  |  |  |  |
| 2 | 203 | 1 | hop 1 | 0.355 | 0.355 | 0.000 |  | 0.645 |
| 2 | 203 | 2 | answer | 0.310 | 0.000 | 0.310 |  | 0.690 |
| 2 | 203 | 3 | answer | 0.320 | 0.000 | 0.320 |  | 0.680 |
| 3 | 206 | 1 | hop 1 | 0.461 | 0.461 | 0.000 | 0.000 | 0.539 |
| 3 | 206 | 2 | hop 2 | 0.286 | 0.000 | 0.286 | 0.000 | 0.714 |
| 3 | 206 | 3 | answer | 0.199 | 0.000 | 0.000 | 0.199 | 0.801 |

`holds hop j`: share of examples whose top prediction at the entity token is hop j's entity (`other`: anything else). Loop 0 is the frozen backbone.
