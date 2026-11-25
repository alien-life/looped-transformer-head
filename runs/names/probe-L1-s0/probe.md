# Per-loop decoding probe

## frozen backbone

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 0 | answer | 0.126 |  |  |  |  |
| 2 | 203 | 0 | answer | 0.010 | 0.015 | 0.044 |  | 0.941 |
| 3 | 206 | 0 | answer | 0.000 | 0.019 | 0.000 | 0.005 | 0.976 |

## 1x-plain-L1-s0 (R=1, final answer only)

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 1 | answer | 0.581 |  |  |  |  |
| 2 | 203 | 1 | answer | 0.340 | 0.000 | 0.340 |  | 0.660 |
| 3 | 206 | 1 | answer | 0.223 | 0.000 | 0.000 | 0.223 | 0.777 |

## 3x-plain-L1-s0 (R=3, final answer only)

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 1 | answer | 0.524 |  |  |  |  |
| 1 | 191 | 2 | answer | 0.565 |  |  |  |  |
| 1 | 191 | 3 | answer | 0.602 |  |  |  |  |
| 2 | 203 | 1 | hop 1 | 0.010 | 0.010 | 0.296 |  | 0.695 |
| 2 | 203 | 2 | answer | 0.300 | 0.000 | 0.300 |  | 0.700 |
| 2 | 203 | 3 | answer | 0.281 | 0.000 | 0.281 |  | 0.719 |
| 3 | 206 | 1 | hop 1 | 0.073 | 0.073 | 0.073 | 0.029 | 0.825 |
| 3 | 206 | 2 | hop 2 | 0.005 | 0.000 | 0.005 | 0.180 | 0.816 |
| 3 | 206 | 3 | answer | 0.189 | 0.000 | 0.000 | 0.189 | 0.811 |

## 2x-chain-L1-s0 (R=2, chain-supervised)

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 1 | answer | 0.555 |  |  |  |  |
| 1 | 191 | 2 | answer | 0.539 |  |  |  |  |
| 2 | 203 | 1 | hop 1 | 0.384 | 0.384 | 0.000 |  | 0.616 |
| 2 | 203 | 2 | answer | 0.300 | 0.000 | 0.300 |  | 0.700 |
| 3 | 206 | 1 | hop 1 | 0.456 | 0.456 | 0.000 | 0.000 | 0.544 |
| 3 | 206 | 2 | answer | 0.214 | 0.000 | 0.000 | 0.214 | 0.786 |

## 3x-chain-L1-s0 (R=3, chain-supervised)

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 1 | answer | 0.571 |  |  |  |  |
| 1 | 191 | 2 | answer | 0.592 |  |  |  |  |
| 1 | 191 | 3 | answer | 0.592 |  |  |  |  |
| 2 | 203 | 1 | hop 1 | 0.360 | 0.360 | 0.000 |  | 0.640 |
| 2 | 203 | 2 | answer | 0.315 | 0.000 | 0.315 |  | 0.685 |
| 2 | 203 | 3 | answer | 0.310 | 0.000 | 0.310 |  | 0.690 |
| 3 | 206 | 1 | hop 1 | 0.466 | 0.466 | 0.000 | 0.000 | 0.534 |
| 3 | 206 | 2 | hop 2 | 0.311 | 0.000 | 0.311 | 0.000 | 0.689 |
| 3 | 206 | 3 | answer | 0.228 | 0.000 | 0.000 | 0.228 | 0.772 |

`holds hop j`: share of examples whose top prediction at the entity token is hop j's entity (`other`: anything else). Loop 0 is the frozen backbone.
